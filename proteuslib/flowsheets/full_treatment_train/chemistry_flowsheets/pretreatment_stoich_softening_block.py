###############################################################################
# ProteusLib Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/nawi-hub/proteuslib/"
#
###############################################################################

"""
    Stoichiometric Softening pretreatment process
    ----------------------------------------------
    This will build a stoichiometric pretreatment process unit by unit. The
    overall process is diagrammed below.

                    Lime stream
                        |
                        V
    inlet stream ---> [Mixer] --- outlet stream ---> [StoichiometricReactor] ---> mixed flow ... (see below)


    ...mixed flow ---> [Separator] ---> exit stream (to RO)
                           |
                           V
                       waste stream

    Stoich Reactions:
         Ca(HCO3)2 +   Ca(OH)2 --> 2 CaCO3 +  2 H2O
         Mg(HCO3)2 + 2 Ca(OH)2 --> 2 CaCO3 + Mg(OH)2 +  2 H2O
"""

# Importing the object for units from pyomo
from pyomo.environ import units as pyunits

# Imports from idaes core
from idaes.core import AqueousPhase
from idaes.core.components import Solvent, Solute, Cation, Anion
from idaes.core.phases import PhaseType as PT

# Imports from idaes generic models
import idaes.generic_models.properties.core.pure.Perrys as Perrys
from idaes.generic_models.properties.core.pure.ConstantProperties import Constant
from idaes.generic_models.properties.core.state_definitions import FTPx
from idaes.generic_models.properties.core.eos.ideal import Ideal
from idaes.generic_models.properties.core.reactions.rate_constant import arrhenius
from idaes.generic_models.properties.core.reactions.rate_forms import power_law_rate

# Importing the enum for concentration unit basis used in the 'get_concentration_term' function
from idaes.generic_models.properties.core.generic.generic_reaction import ConcentrationForm

# Import the object/function for heat of reaction
from idaes.generic_models.properties.core.reactions.dh_rxn import constant_dh_rxn

# Import safe log power law equation
from idaes.generic_models.properties.core.reactions.equilibrium_forms import log_power_law_equil

# Import built-in van't Hoff function
from idaes.generic_models.properties.core.reactions.equilibrium_constant import van_t_hoff

# Import specific pyomo objects
from pyomo.environ import (ConcreteModel,
                           Var,
                           Constraint,
                           SolverStatus,
                           TerminationCondition,
                           TransformationFactory,
                           value,
                           Suffix)

from pyomo.network import Arc

from idaes.core.util import scaling as iscale
from idaes.core.util.initialization import fix_state_vars, revert_state_vars

# Import pyomo methods to check the system units
from pyomo.util.check_units import assert_units_consistent


from proteuslib.flowsheets.full_treatment_train.util import solve_with_user_scaling, check_dof
from proteuslib.flowsheets.full_treatment_train.example_models import property_models
from idaes.core.util import get_solver

# Import the idaes objects for Generic Properties and Reactions
from idaes.generic_models.properties.core.generic.generic_property import (
        GenericParameterBlock)
from idaes.generic_models.properties.core.generic.generic_reaction import (
        GenericReactionParameterBlock)

# Import the idaes object for the StoichiometricReactor unit model
from idaes.generic_models.unit_models.stoichiometric_reactor import \
    StoichiometricReactor

# Import th idaes object for the Mixer and Separator unit model
from idaes.generic_models.unit_models.mixer import Mixer
from idaes.generic_models.unit_models import Separator

from idaes.generic_models.unit_models.translator import Translator

# Import the core idaes objects for Flowsheets and types of balances
from idaes.core import FlowsheetBlock

# Import log10 function from pyomo
from pyomo.environ import log10

import idaes.logger as idaeslog

# Grab the scaling utilities
from proteuslib.flowsheets.full_treatment_train.electrolyte_scaling_utils import (
    approximate_chemical_state_args,
    calculate_chemical_scaling_factors,
    calculate_chemical_scaling_factors_for_energy_balances)

from proteuslib.flowsheets.full_treatment_train.chemical_flowsheet_util import (
    set_H2O_molefraction, zero_out_non_H2O_molefractions, fix_all_molefractions,
    unfix_all_molefractions, seq_decomp_initializer )

__author__ = "Austin Ladshaw, Srikanth Allu"

# Configuration dictionary
stoich_softening_thermo_config = {
    "components": {
        'H2O': {"type": Solvent,
              # Define the methods used to calculate the following properties
              "dens_mol_liq_comp": Perrys,
              "enth_mol_liq_comp": Perrys,
              "cp_mol_liq_comp": Perrys,
              "entr_mol_liq_comp": Perrys,
              # Parameter data is always associated with the methods defined above
              "parameter_data": {
                    "mw": (18.0153, pyunits.g/pyunits.mol),
                    "pressure_crit": (220.64E5, pyunits.Pa),
                    "temperature_crit": (647, pyunits.K),
                    # Comes from Perry's Handbook:  p. 2-98
                    "dens_mol_liq_comp_coeff": {
                        '1': (5.459, pyunits.kmol*pyunits.m**-3),
                        '2': (0.30542, pyunits.dimensionless),
                        '3': (647.13, pyunits.K),
                        '4': (0.081, pyunits.dimensionless)},
                    "enth_mol_form_liq_comp_ref": (-285.830, pyunits.kJ/pyunits.mol),
                    "enth_mol_form_vap_comp_ref": (0, pyunits.kJ/pyunits.mol),
                    # Comes from Perry's Handbook:  p. 2-174
                    "cp_mol_liq_comp_coeff": {
                        '1': (2.7637E5, pyunits.J/pyunits.kmol/pyunits.K),
                        '2': (-2.0901E3, pyunits.J/pyunits.kmol/pyunits.K**2),
                        '3': (8.125, pyunits.J/pyunits.kmol/pyunits.K**3),
                        '4': (-1.4116E-2, pyunits.J/pyunits.kmol/pyunits.K**4),
                        '5': (9.3701E-6, pyunits.J/pyunits.kmol/pyunits.K**5)},
                    "cp_mol_ig_comp_coeff": {
                        'A': (30.09200, pyunits.J/pyunits.mol/pyunits.K),
                        'B': (6.832514, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-1),
                        'C': (6.793435, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-2),
                        'D': (-2.534480, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-3),
                        'E': (0.082139, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**2),
                        'F': (-250.8810, pyunits.kJ/pyunits.mol),
                        'G': (223.3967, pyunits.J/pyunits.mol/pyunits.K),
                        'H': (0, pyunits.kJ/pyunits.mol)},
                    "entr_mol_form_liq_comp_ref": (69.95, pyunits.J/pyunits.K/pyunits.mol),
                    "pressure_sat_comp_coeff": {
                        'A': (4.6543, None),  # [1], temperature range 255.9 K - 373 K
                        'B': (1435.264, pyunits.K),
                        'C': (-64.848, pyunits.K)}
                                },
                    # End parameter_data
                    },
        'Ca(OH)2': {  "type": Solute,  "valid_phase_types": PT.aqueousPhase,
                    # Define the methods used to calculate the following properties
                    "dens_mol_liq_comp": Constant,
                    "enth_mol_liq_comp": Constant,
                    "cp_mol_liq_comp": Constant,
                    "entr_mol_liq_comp": Constant,
                    # Parameter data is always associated with the methods defined above
                    "parameter_data": {
                        "mw": (74.093, pyunits.g/pyunits.mol),
                        "dens_mol_liq_comp_coeff": (55, pyunits.kmol*pyunits.m**-3),
                        "enth_mol_form_liq_comp_ref": (-945.53, pyunits.kJ/pyunits.mol),
                        "cp_mol_liq_comp_coeff": (167039, pyunits.J/pyunits.kmol/pyunits.K),
                        "entr_mol_form_liq_comp_ref": (100, pyunits.J/pyunits.K/pyunits.mol)
                            },
                    # End parameter_data
                    },
        'NaCl': {  "type": Solute,  "valid_phase_types": PT.aqueousPhase,
                    # Define the methods used to calculate the following properties
                    "dens_mol_liq_comp": Constant,
                    "enth_mol_liq_comp": Constant,
                    "cp_mol_liq_comp": Constant,
                    "entr_mol_liq_comp": Constant,
                    # Parameter data is always associated with the methods defined above
                    "parameter_data": {
                        "mw": (58.44, pyunits.g/pyunits.mol),
                        "dens_mol_liq_comp_coeff": (55, pyunits.kmol*pyunits.m**-3),
                        "enth_mol_form_liq_comp_ref": (-945.53, pyunits.kJ/pyunits.mol),
                        "cp_mol_liq_comp_coeff": (167039, pyunits.J/pyunits.kmol/pyunits.K),
                        "entr_mol_form_liq_comp_ref": (100, pyunits.J/pyunits.K/pyunits.mol)
                            },
                    # End parameter_data
                    },
        'CaCO3': {  "type": Solute,  "valid_phase_types": PT.aqueousPhase,
                    # Define the methods used to calculate the following properties
                    "dens_mol_liq_comp": Constant,
                    "enth_mol_liq_comp": Constant,
                    "cp_mol_liq_comp": Constant,
                    "entr_mol_liq_comp": Constant,
                    # Parameter data is always associated with the methods defined above
                    "parameter_data": {
                        "dens_mol_liq_comp_coeff": (55, pyunits.kmol*pyunits.m**-3),
                        "enth_mol_form_liq_comp_ref": (-945.53, pyunits.kJ/pyunits.mol),
                        "cp_mol_liq_comp_coeff": (167039, pyunits.J/pyunits.kmol/pyunits.K),
                        "entr_mol_form_liq_comp_ref": (100, pyunits.J/pyunits.K/pyunits.mol)
                            },
                    # End parameter_data
                    },
        'Ca(HCO3)2': {  "type": Solute,  "valid_phase_types": PT.aqueousPhase,
                    # Define the methods used to calculate the following properties
                    "dens_mol_liq_comp": Constant,
                    "enth_mol_liq_comp": Constant,
                    "cp_mol_liq_comp": Constant,
                    "entr_mol_liq_comp": Constant,
                    # Parameter data is always associated with the methods defined above
                    "parameter_data": {
                        "dens_mol_liq_comp_coeff": (55, pyunits.kmol*pyunits.m**-3),
                        "enth_mol_form_liq_comp_ref": (-945.53, pyunits.kJ/pyunits.mol),
                        "cp_mol_liq_comp_coeff": (167039, pyunits.J/pyunits.kmol/pyunits.K),
                        "entr_mol_form_liq_comp_ref": (100, pyunits.J/pyunits.K/pyunits.mol)
                            },
                    # End parameter_data
                    },
        'Mg(OH)2': {  "type": Solute,  "valid_phase_types": PT.aqueousPhase,
                    # Define the methods used to calculate the following properties
                    "dens_mol_liq_comp": Constant,
                    "enth_mol_liq_comp": Constant,
                    "cp_mol_liq_comp": Constant,
                    "entr_mol_liq_comp": Constant,
                    # Parameter data is always associated with the methods defined above
                    "parameter_data": {
                        "mw": (74.093, pyunits.g/pyunits.mol),
                        "dens_mol_liq_comp_coeff": (55, pyunits.kmol*pyunits.m**-3),
                        "enth_mol_form_liq_comp_ref": (-945.53, pyunits.kJ/pyunits.mol),
                        "cp_mol_liq_comp_coeff": (167039, pyunits.J/pyunits.kmol/pyunits.K),
                        "entr_mol_form_liq_comp_ref": (100, pyunits.J/pyunits.K/pyunits.mol)
                            },
                    # End parameter_data
                    },
        'Mg(HCO3)2': {  "type": Solute,  "valid_phase_types": PT.aqueousPhase,
                    # Define the methods used to calculate the following properties
                    "dens_mol_liq_comp": Constant,
                    "enth_mol_liq_comp": Constant,
                    "cp_mol_liq_comp": Constant,
                    "entr_mol_liq_comp": Constant,
                    # Parameter data is always associated with the methods defined above
                    "parameter_data": {
                        "dens_mol_liq_comp_coeff": (55, pyunits.kmol*pyunits.m**-3),
                        "enth_mol_form_liq_comp_ref": (-945.53, pyunits.kJ/pyunits.mol),
                        "cp_mol_liq_comp_coeff": (167039, pyunits.J/pyunits.kmol/pyunits.K),
                        "entr_mol_form_liq_comp_ref": (100, pyunits.J/pyunits.K/pyunits.mol)
                            },
                    # End parameter_data
                    },
              },
              # End Component list
        "phases":  {'Liq': {"type": AqueousPhase,
                            "equation_of_state": Ideal},
                    },

        "state_definition": FTPx,
        "state_bounds": {"flow_mol": (0, 50, 100),
                         "temperature": (273.15, 300, 650),
                         "pressure": (5e4, 1e5, 1e6)
                     },
        "pressure_ref": 1e5,
        "temperature_ref": 300,
        "base_units": {"time": pyunits.s,
                       "length": pyunits.m,
                       "mass": pyunits.kg,
                       "amount": pyunits.mol,
                       "temperature": pyunits.K},

    }
    # End softening_thermo_config definition

# This config is for stoich softening
stoich_softening_reaction_config = {
    "base_units": {"time": pyunits.s,
                   "length": pyunits.m,
                   "mass": pyunits.kg,
                   "amount": pyunits.mol,
                   "temperature": pyunits.K},
    "rate_reactions": {
        "R1": {"stoichiometry": {("Liq", "Ca(HCO3)2"): -1,
                                 ("Liq", "Ca(OH)2"): -1,
                                 ("Liq", "CaCO3"): 2,
                                 ("Liq", "H2O"): 2},
               "heat_of_reaction": constant_dh_rxn,
               "rate_constant" : arrhenius,
               "rate_form" : power_law_rate,
               "concentration_form" : ConcentrationForm.moleFraction,
               "parameter_data": {
                   "arrhenius_const" : (1, pyunits.mol/pyunits.m**3/pyunits.s),
                   "energy_activation" : (0, pyunits.J/pyunits.mol),
                   "dh_rxn_ref": (0, pyunits.J/pyunits.mol)
              }
         },
        "R2": {"stoichiometry": {("Liq", "Mg(HCO3)2"): -1,
                                 ("Liq", "Ca(OH)2"): -2,
                                 ("Liq", "CaCO3"): 2,
                                 ("Liq", "Mg(OH)2"): 1,
                                 ("Liq", "H2O"): 2},
               "heat_of_reaction": constant_dh_rxn,
               "rate_constant" : arrhenius,
               "rate_form" : power_law_rate,
               "concentration_form" : ConcentrationForm.moleFraction,
               "parameter_data": {
                   "arrhenius_const" : (1, pyunits.mol/pyunits.m**3/pyunits.s),
                   "energy_activation" : (0, pyunits.J/pyunits.mol),
                   "dh_rxn_ref": (0, pyunits.J/pyunits.mol)
              }
         }
    }
}
# End reaction_config definition

# Get default solver for testing
solver = get_solver()

def build_stoich_softening_prop(model):
    model.fs.stoich_softening_thermo_params = GenericParameterBlock(default=stoich_softening_thermo_config)
    model.fs.stoich_softening_rxn_params = GenericReactionParameterBlock(
            default={"property_package": model.fs.stoich_softening_thermo_params, **stoich_softening_reaction_config})

def build_stoich_softening_mixer_unit(model):
    model.fs.stoich_softening_mixer_unit = Mixer(default={
            "property_package": model.fs.stoich_softening_thermo_params,
            "inlet_list": ["inlet_stream", "lime_stream"]})

    # add new constraint for dosing rate
    dr = model.fs.stoich_softening_mixer_unit.lime_stream.flow_mol[0].value* \
            model.fs.stoich_softening_mixer_unit.lime_stream.mole_frac_comp[0, "Ca(OH)2"].value
    dr = dr*74.093*1000
    model.fs.stoich_softening_mixer_unit.dosing_rate = Var(initialize=dr)

    def _dosing_rate_cons(blk):
        return blk.dosing_rate == blk.lime_stream.flow_mol[0]*blk.lime_stream.mole_frac_comp[0, "Ca(OH)2"]*74.093*1000

    model.fs.stoich_softening_mixer_unit.dosing_cons = Constraint( rule=_dosing_rate_cons )

def set_stoich_softening_mixer_inlets(model, dosing_rate_of_lime_mg_per_s = 1,
                                        inlet_water_density_kg_per_L = 1,
                                        inlet_temperature_K = 298,
                                        inlet_pressure_Pa = 101325,
                                        inlet_flow_mol_per_s = 10,
                                        inlet_total_hardness_mg_per_L=200,
                                        hardness_fraction_to_Ca=0.5,
                                        inlet_salinity_psu=35):

    #inlet stream
    model.fs.stoich_softening_mixer_unit.inlet_stream.flow_mol[0].set_value(inlet_flow_mol_per_s)
    model.fs.stoich_softening_mixer_unit.inlet_stream.pressure[0].set_value(inlet_pressure_Pa)
    model.fs.stoich_softening_mixer_unit.inlet_stream.temperature[0].set_value(inlet_temperature_K)

    zero_out_non_H2O_molefractions(model.fs.stoich_softening_mixer_unit.inlet_stream)
    # Calculate molefractions for Ca(HCO3)2 and Mg(HCO3)2
    total_molar_density = inlet_water_density_kg_per_L/18*1000 #mol/L
    if hardness_fraction_to_Ca > 1:
        hardness_fraction_to_Ca = 1
    if hardness_fraction_to_Ca < 0:
        hardness_fraction_to_Ca = 0
    x_Ca = (inlet_total_hardness_mg_per_L/50000/2/total_molar_density)*hardness_fraction_to_Ca
    x_Mg = (inlet_total_hardness_mg_per_L/50000/2/total_molar_density)*(1-hardness_fraction_to_Ca)
    model.fs.stoich_softening_mixer_unit.inlet_stream.mole_frac_comp[0, "Ca(HCO3)2"].set_value(x_Ca)
    model.fs.stoich_softening_mixer_unit.inlet_stream.mole_frac_comp[0, "Mg(HCO3)2"].set_value(x_Mg)

    total_salt = value(model.fs.stoich_softening_mixer_unit.inlet_stream.mole_frac_comp[0, "Ca(HCO3)2"])*total_molar_density*101
    total_salt += value(model.fs.stoich_softening_mixer_unit.inlet_stream.mole_frac_comp[0, "Mg(HCO3)2"])*total_molar_density*85.31
    psu_from_hardness = total_salt/(total_molar_density*18)*1000
    if psu_from_hardness < inlet_salinity_psu:
        psu_from_nacl = inlet_salinity_psu-psu_from_hardness
        x_NaCl = psu_from_nacl*(total_molar_density*18)/1000/total_molar_density/58.44
        model.fs.stoich_softening_mixer_unit.inlet_stream.mole_frac_comp[0, "NaCl"].set_value(x_NaCl)

    set_H2O_molefraction(model.fs.stoich_softening_mixer_unit.inlet_stream)

    #lime stream
    model.fs.stoich_softening_mixer_unit.lime_stream.pressure[0].set_value(inlet_pressure_Pa)
    model.fs.stoich_softening_mixer_unit.lime_stream.temperature[0].set_value(inlet_temperature_K)
    # Use given dosing rate value to estimate OCl_- molefraction and flow rate for naocl stream
    zero_out_non_H2O_molefractions(model.fs.stoich_softening_mixer_unit.lime_stream)
    model.fs.stoich_softening_mixer_unit.lime_stream.mole_frac_comp[0, "Ca(OH)2"].set_value(1)
    set_H2O_molefraction(model.fs.stoich_softening_mixer_unit.lime_stream)
    flow_of_lime = dosing_rate_of_lime_mg_per_s/ \
                model.fs.stoich_softening_mixer_unit.lime_stream.mole_frac_comp[0, "Ca(OH)2"].value/ \
                74.44/1000
    model.fs.stoich_softening_mixer_unit.lime_stream.flow_mol[0].set_value(flow_of_lime)

    model.fs.stoich_softening_mixer_unit.dosing_rate.set_value(dosing_rate_of_lime_mg_per_s)

def fix_stoich_softening_mixer_inlets(model):
    model.fs.stoich_softening_mixer_unit.inlet_stream.flow_mol[0].fix()
    model.fs.stoich_softening_mixer_unit.inlet_stream.pressure[0].fix()
    model.fs.stoich_softening_mixer_unit.inlet_stream.temperature[0].fix()
    fix_all_molefractions(model.fs.stoich_softening_mixer_unit.inlet_stream)

    model.fs.stoich_softening_mixer_unit.lime_stream.flow_mol[0].fix()
    model.fs.stoich_softening_mixer_unit.lime_stream.pressure[0].fix()
    model.fs.stoich_softening_mixer_unit.lime_stream.temperature[0].fix()
    fix_all_molefractions(model.fs.stoich_softening_mixer_unit.lime_stream)

def unfix_stoich_softening_mixer_inlet_stream(model):
    model.fs.stoich_softening_mixer_unit.inlet_stream.flow_mol[0].unfix()
    model.fs.stoich_softening_mixer_unit.inlet_stream.pressure[0].unfix()
    model.fs.stoich_softening_mixer_unit.inlet_stream.temperature[0].unfix()
    unfix_all_molefractions(model.fs.stoich_softening_mixer_unit.inlet_stream)

def unfix_stoich_softening_mixer_lime_stream(model):
    model.fs.stoich_softening_mixer_unit.lime_stream.flow_mol[0].unfix()
    model.fs.stoich_softening_mixer_unit.lime_stream.pressure[0].unfix()
    model.fs.stoich_softening_mixer_unit.lime_stream.temperature[0].unfix()
    unfix_all_molefractions(model.fs.ideal_naocl_mixer_unit.lime_stream)

def scale_stoich_softening_mixer(unit):
    iscale.constraint_autoscale_large_jac(unit)

def initialize_stoich_softening_mixer(unit, debug_out=False):
    solver.options['bound_push'] = 1e-10
    solver.options['mu_init'] = 1e-6
    solver.options["nlp_scaling_method"] = "user-scaling"
    was_fixed = False
    if unit.lime_stream.flow_mol[0].is_fixed() == False:
        unit.lime_stream.flow_mol[0].fix()
        was_fixed = True
    if debug_out == True:
        unit.initialize(optarg=solver.options, outlvl=idaeslog.DEBUG)
    else:
        unit.initialize(optarg=solver.options)
    if was_fixed == True:
        unit.lime_stream.flow_mol[0].unfix()

def display_results_of_stoich_softening_mixer(unit):
    print()
    print("=========== Stoich Softening Mixer Results ============")
    print("Outlet Temperature:       \t" + str(unit.outlet.temperature[0].value))
    print("Outlet Pressure:          \t" + str(unit.outlet.pressure[0].value))
    print("Outlet FlowMole:          \t" + str(unit.outlet.flow_mol[0].value))
    print()
    total_molar_density = 55.6
    total_salt = value(unit.outlet.mole_frac_comp[0, "Ca(OH)2"])*total_molar_density*74.093
    total_salt += value(unit.outlet.mole_frac_comp[0, "Ca(HCO3)2"])*total_molar_density*101
    total_salt += value(unit.outlet.mole_frac_comp[0, "Mg(HCO3)2"])*total_molar_density*85.31
    total_salt += value(unit.outlet.mole_frac_comp[0, "CaCO3"])*total_molar_density*100
    total_salt += value(unit.outlet.mole_frac_comp[0, "Mg(OH)2"])*total_molar_density*58.32
    total_salt += value(unit.outlet.mole_frac_comp[0, "NaCl"])*total_molar_density*58.44
    psu = total_salt/(total_molar_density*18)*1000
    print("STP Salinity (PSU):           \t" + str(psu))
    print("Lime Dosing Rate (mg/s): \t" + str(unit.dosing_rate.value))
    print("------------------------------------------------------")
    print()

def run_stoich_softening_mixer_example():
    model = ConcreteModel()
    model.fs = FlowsheetBlock(default={"dynamic": False})

    # Add properties to model
    build_stoich_softening_prop(model)

    # add the mixer
    build_stoich_softening_mixer_unit(model)

    # set model values
    set_stoich_softening_mixer_inlets(model)

    model.fs.stoich_softening_mixer_unit.lime_stream.mole_frac_comp.pprint()
    model.fs.stoich_softening_mixer_unit.inlet_stream.mole_frac_comp.pprint()
    model.fs.stoich_softening_mixer_unit.dosing_rate.pprint()

    # fix inlets for testing
    fix_stoich_softening_mixer_inlets(model)

    check_dof(model)

    # scale the mixer
    scale_stoich_softening_mixer(model.fs.stoich_softening_mixer_unit)

    # initializer mixer
    initialize_stoich_softening_mixer(model.fs.stoich_softening_mixer_unit, debug_out=False)

    # solve with user scaling
    solve_with_user_scaling(model, tee=True, bound_push=1e-10, mu_init=1e-6)

    model.fs.stoich_softening_mixer_unit.outlet.mole_frac_comp.pprint()
    model.fs.stoich_softening_mixer_unit.dosing_rate.pprint()

    display_results_of_stoich_softening_mixer(model.fs.stoich_softening_mixer_unit)

    return model

if __name__ == "__main__":
    model = run_stoich_softening_mixer_example()