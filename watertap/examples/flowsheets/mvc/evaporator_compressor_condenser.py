###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################
from pyomo.environ import (
    ConcreteModel,
    value,
    Constraint,
    Expression,
    Objective,
    Param,
    TransformationFactory,
    units as pyunits,
    assert_optimal_termination,
)
from pyomo.network import Arc

from pyomo.util.check_units import assert_units_consistent
from idaes.core import FlowsheetBlock
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.unit_models.mvc.components import Evaporator, Compressor, Condenser
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w


def main():
    # build
    m = build()
    set_operating_conditions(m)
    initialize_system(m)
    # m.fs.evaporator.connect_to_condenser(m.fs.condenser)
    print(degrees_of_freedom(m))

    solver = get_solver()
    results = solver.solve(m, tee=False)
    assert_optimal_termination(results)

    print('Evaporator temperature: ', m.fs.evaporator.properties_brine[0].temperature)
    print('Evaporator pressure: ', m.fs.evaporator.properties_brine[0].pressure)
    print('Compressor outlet temperature: ', m.fs.compressor.outlet[0].temperature)
    print('Compressor outlet pressure: ', m.fs.compressor.outlet[0].pressure)
    print('Condenser outlet temperature: ', m.fs.condenser.outlet[0].temperature)
    print('Condenser outlet temperature: ', m.fs.compressor.outlet[0].pressure)


def build():
    # flowsheet set up
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Properties
    m.fs.properties_feed = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()

    # Unit models
    m.fs.evaporator = Evaporator(
        default={
            "property_package_feed": m.fs.properties_feed,
            "property_package_vapor": m.fs.properties_vapor,
        }
    )

    m.fs.compressor = Compressor(default={"property_package": m.fs.properties_vapor})

    m.fs.condenser = Condenser(default={"property_package": m.fs.properties_vapor})

    # Connections
    m.fs.s01 = Arc(source=m.fs.evaporator.outlet_vapor, destination=m.fs.compressor.inlet)
    m.fs.s02 = Arc(source=m.fs.compressor.outlet, destination=m.fs.condenser.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)
    m.fs.evaporator.connect_to_condenser(m.fs.condenser)
    # Scaling
    # properties
    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )
    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "TDS")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Vap", "H2O")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )

    # unit model values
    # evaporator
    iscale.set_scaling_factor(m.fs.evaporator.area, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.U, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_in, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_out, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.lmtd, 1e-1)
    # iscale.set_scaling_factor(m.fs.evaporator.heat_transfer, 1e-6)

    # compressor
    iscale.set_scaling_factor(m.fs.compressor.control_volume.work, 1e-6)

    # condenser
    iscale.set_scaling_factor(m.fs.condenser.control_volume.heat, 1e-6)

    # calculate and propagate scaling factors
    iscale.calculate_scaling_factors(m)

    return m

def set_operating_conditions(m):
    # evaporator feed inlet
    m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(10)
    m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0.05)
    m.fs.evaporator.inlet_feed.temperature[0].fix(273.15 + 50.52)  # K
    m.fs.evaporator.inlet_feed.pressure[0].fix(1e5)  # Pa

    # evaporator specifications
    m.fs.evaporator.outlet_brine.temperature[0].fix(273.15 + 60)
    m.fs.evaporator.U.fix(1e3)  # W/K-m^2
    m.fs.evaporator.area.fix(100)  # m^2

    # compressor
    m.fs.compressor.pressure_ratio.fix(2)
    m.fs.compressor.efficiency.fix(0.8)

    # check degrees of freedom
    print(degrees_of_freedom(m))

def initialize_system(m, solver=None):
    if solver is None:
        solver = get_solver()
    optarg = solver.options

    # initialize evaporator
    #m.fs.evaporator.connection_to_condenser.deactivate()
    #m.fs.evaporator.delta_temperature_out.fix(5)
    #m.fs.evaporator.delta_temperature_in.fix(30)
    m.fs.evaporator.initialize_build(delta_temperature_in=30, delta_temperature_out=5)
    #m.fs.evaporator.connection_to_condenser.activate()
    #m.fs.evaporator.delta_temperature_out.unfix()
    #m.fs.evaporator.delta_temperature_in.unfix()

    # m.fs.evaporator.initialize(delta_temperature_in=30, delta_temperature_out=5)
    # keyword arguments - fix those values, initialize, unfix those values
    # if user doesn't provide the keyword arguments, check to see if they are fixed already, if theyare just initialize
    # if not Raise an error - you must specify the delta temperature in order to initialize

    # initialize compressor
    propagate_state(m.fs.s01)
    m.fs.compressor.initialize_build()

    # initialize condenser
    propagate_state(m.fs.s02)
    m.fs.condenser.initialize_build(heat=-m.fs.evaporator.heat_transfer.value)

    # match heat transfer from evaporator to heat transfer out to have zero DOF
    # m.fs.condenser.initialize(heat=-m.fs.evaporator.heat.value)
    # keyword argument fix the heat transfer, initialize, unfix the heat transfer
    # if user doesn't provide, check that it is fixed and True: initialize
    # unfix heat transfer

if __name__ == "__main__":
    main()