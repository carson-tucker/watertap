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
from pyomo.environ import ConcreteModel, assert_optimal_termination, value
from pyomo.util.check_units import assert_units_consistent
from idaes.core import FlowsheetBlock
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.unit_models.mvc.components import Evaporator, Condenser
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w


def main():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.properties_feed = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()


    # Unit models
    m.fs.evaporator = Evaporator(
        default={
            "property_package_feed": m.fs.properties_feed,
            "property_package_vapor": m.fs.properties_vapor,
        }
    )
    m.fs.condenser = Condenser(default={"property_package": m.fs.properties_vapor})
    m.fs.evaporator.connect_to_condenser(m.fs.condenser)


    # scaling
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
    # scaling - evaporator
    iscale.set_scaling_factor(m.fs.evaporator.area, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.U, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_in, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_out, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.lmtd, 1e-1)
    # iscale.set_scaling_factor(m.fs.evaporator.heat_transfer, 1e-6)
    # scaling - condenser
    iscale.set_scaling_factor(m.fs.condenser.control_volume.heat, 1e-6)
    # scaling - calculate and propagate
    iscale.calculate_scaling_factors(m)


    # state variables
    # Feed inlet
    m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(10)
    m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0.5)
    m.fs.evaporator.inlet_feed.temperature[0].fix(273.15 + 50.52)  # K
    m.fs.evaporator.inlet_feed.pressure[0].fix(1e5)  # Pa

    # Condenser inlet
    m.fs.condenser.inlet.flow_mass_phase_comp[0, "Vap", "H2O"].fix(1)
    m.fs.condenser.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(1e-8)
    #m.fs.condenser.inlet.temperature[0].fix(400)  # K
    m.fs.condenser.inlet.pressure[0].fix(0.5e5)  # Pa

    # Evaporator/condenser specifications
    m.fs.evaporator.outlet_brine.temperature[0].fix(273.15 + 60)
    m.fs.evaporator.U.fix(1e3)  # W/K-m^2
    #m.fs.evaporator.area.fix(100)  # m^2
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp['Vap','H2O'].fix(1)
    m.fs.evaporator.delta_temperature_in.fix(66.85)
    #m.fs.evaporator.delta_temperature_out.fix(0.8236)

    # initialize
    #m.fs.evaporator.initialize_build(delta_temperature_in=30, delta_temperature_out=5)
    m.fs.evaporator.initialize_build()
    m.fs.condenser.initialize_build(heat=-m.fs.evaporator.heat_transfer.value)

    # check build
    assert_units_consistent(m)
    print(degrees_of_freedom(m))
    # assert degrees_of_freedom(m) == 0 # will equal 2 because of LMTD variables needed

    # solve
    solver = get_solver()
    results = solver.solve(m, tee=False)
    assert_optimal_termination(results)
    recovery = m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp['Vap','H2O'].value/(m.fs.evaporator.properties_feed[0].flow_mass_phase_comp['Liq','TDS'].value + m.fs.evaporator.properties_feed[0].flow_mass_phase_comp['Liq','H2O'].value)
    print('Evaporator heat transfer: ', m.fs.evaporator.heat_transfer.value)
    print('Feed inlet TDS mass frac: ', m.fs.evaporator.properties_feed[0].mass_frac_phase_comp['Liq','TDS'].value)
    print('Feed inlet enth_flow: ', value(m.fs.evaporator.properties_feed[0].enth_flow))
    print('Brine inlet enth_flow: ', value(m.fs.evaporator.properties_brine[0].enth_flow))
    print('Vapor inlet enth_flow: ', m.fs.evaporator.properties_vapor[0].enth_flow_phase['Vap'].value)
    print('Recovery: ', recovery)
    print("Vapor flow rate:", m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp['Vap','H2O'].value)
    print("Evaporator Temperature: ",  m.fs.evaporator.properties_brine[0].temperature.value)
    print("Evaporator Pressure: ",  m.fs.evaporator.properties_brine[0].pressure.value)
    print("Heat transfer: ",  m.fs.evaporator.heat_transfer.value)
    print("LMTD: ",  m.fs.evaporator.lmtd.value)
    print("Evaporator area: ", m.fs.evaporator.area.value)

if __name__ == "__main__":
    main()