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
    Var,
    TransformationFactory,
    units as pyunits,
    assert_optimal_termination,
)
from pyomo.network import Arc

from pyomo.util.check_units import assert_units_consistent
import pyomo.util.infeasible as infeas
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_diagnostics import DegeneracyHunter
from idaes.models.unit_models import Feed, Separator, Mixer, Product
from idaes.models.unit_models.translator import Translator
from idaes.models.unit_models.mixer import MomentumMixingType
from idaes.models.unit_models.heat_exchanger import (
    HeatExchanger,
    HeatExchangerFlowPattern,
)
from idaes.generic_models.costing import UnitModelCostingBlock
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.unit_models.mvc.components import Evaporator, Compressor, Condenser
from watertap.unit_models.mvc.components.lmtd_chen_callback import (
    delta_temperature_chen_callback,
)
from watertap.unit_models.pressure_changer import Pump
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w
from watertap.costing import WaterTAPCosting, PumpType


def main():
    # build
    m = build()
    set_operating_conditions(m)
    initialize_system(m)
    solver = get_solver()
    m.fs.objective = Objective(expr=1)
    results = solver.solve(m, tee=False)
    print('First solve')
    print(results.solver.termination_condition)
    if results.solver.termination_condition == "infeasible":
        debug_infeasible(m.fs, solver)

    m.fs.hx_distillate.area.unfix()
    m.fs.evaporator.inlet_feed.temperature[0].unfix()
    results = solver.solve(m, tee=False)
    print('Second solve')
    print(results.solver.termination_condition)
    print('Preheated feed temperature: ', m.fs.evaporator.inlet_feed.temperature[0].value)
    print('Preheater area: ', m.fs.hx_distillate.area.value)

    if results.solver.termination_condition == "infeasible":
        debug_infeasible(m.fs, solver)

    assert False

    assert_optimal_termination(results)
    display_system(m)
    print("\nPreheated Feed State:")
    display_seawater_states(m.fs.evaporator.properties_feed[0])
    print("\nVapor State:")
    display_water_states(m.fs.evaporator.properties_vapor[0])
    print("\nIsentropic compressed Vapor State:")
    display_water_states(m.fs.compressor.properties_isentropic_out[0])
    print("\nCompressed Vapor State:")
    display_water_states(m.fs.compressor.control_volume.properties_out[0])
    print("\nCondensed Vapor State:")
    display_water_states(m.fs.condenser.control_volume.properties_out[0])
    print("\nDistillate exiting HX State:")
    display_water_states(m.fs.distillate.properties[0])
    m.fs.condenser.report()
    m.fs.pump_distillate.report()
    m.fs.hx_distillate.report()
    # optimize(m)
    # display_system(m)


def build():
    # flowsheet set up
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Properties
    m.fs.properties_feed = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()

    # Unit models
    m.fs.feed = Feed(default={"property_package": m.fs.properties_feed})
    m.fs.pump_feed = Pump(default={"property_package": m.fs.properties_feed})

    m.fs.hx_distillate = HeatExchanger(
        default={
            "hot_side_name": "hot",
            "cold_side_name": "cold",
            "hot": {"property_package": m.fs.properties_feed},
            "cold": {"property_package": m.fs.properties_feed},
            "delta_temperature_callback": delta_temperature_chen_callback,
            "flow_pattern": HeatExchangerFlowPattern.countercurrent,
        }
    )
    add_pressure_drop_to_hx(m.fs.hx_distillate, m.fs.config.time)

    m.fs.evaporator = Evaporator(
        default={
            "property_package_feed": m.fs.properties_feed,
            "property_package_vapor": m.fs.properties_vapor,
        }
    )

    m.fs.compressor = Compressor(default={"property_package": m.fs.properties_vapor})

    m.fs.condenser = Condenser(default={"property_package": m.fs.properties_vapor})

    m.fs.tb_distillate = Translator(
        default={
            "inlet_property_package": m.fs.properties_vapor,
            "outlet_property_package": m.fs.properties_feed
        }
    )

    # Translator block to convert distillate exiting condenser from water to seawater prop pack
    @m.fs.tb_distillate.Constraint()
    def eq_flow_mass_comp(blk):
        return (
                blk.properties_in[0].flow_mass_phase_comp['Liq', 'H2O']
                == blk.properties_out[0].flow_mass_phase_comp["Liq", 'H2O']
        )

    @m.fs.tb_distillate.Constraint()
    def eq_temperature(blk):
        return (
                blk.properties_in[0].temperature
                == blk.properties_out[0].temperature
        )

    @m.fs.tb_distillate.Constraint()
    def eq_pressure(blk):
        return (
                blk.properties_in[0].pressure
                == blk.properties_out[0].pressure
        )

    m.fs.pump_brine = Pump(default={"property_package": m.fs.properties_feed})

    m.fs.pump_distillate = Pump(default={"property_package": m.fs.properties_feed})

    m.fs.distillate = Product(default={"property_package": m.fs.properties_feed})

    m.fs.brine = Product(default={"property_package": m.fs.properties_feed})

    # Connections
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.pump_feed.inlet)
    m.fs.s02 = Arc(
        source=m.fs.pump_feed.outlet, destination=m.fs.hx_distillate.cold_inlet
    )
    m.fs.s03 = Arc(
        source=m.fs.hx_distillate.cold_outlet, destination=m.fs.evaporator.inlet_feed
    )
    m.fs.s04 = Arc(
        source=m.fs.evaporator.outlet_vapor, destination=m.fs.compressor.inlet
    )
    m.fs.s05 = Arc(source=m.fs.compressor.outlet, destination=m.fs.condenser.inlet)
    m.fs.s06 = Arc(
        source=m.fs.evaporator.outlet_brine, destination=m.fs.pump_brine.inlet
    )
    m.fs.s07 = Arc(source=m.fs.pump_brine.outlet, destination=m.fs.brine.inlet)
    m.fs.s08 = Arc(source=m.fs.condenser.outlet, destination=m.fs.tb_distillate.inlet)
    m.fs.s09 = Arc(source=m.fs.tb_distillate.outlet, destination=m.fs.pump_distillate.inlet)
    m.fs.s10 = Arc(
        source=m.fs.pump_distillate.outlet, destination=m.fs.hx_distillate.hot_inlet
    )
    m.fs.s11 = Arc(
        source=m.fs.hx_distillate.hot_outlet, destination=m.fs.distillate.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)
    m.fs.evaporator.connect_to_condenser(m.fs.condenser)

    # Add costing
    #add_costing(m)

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
    # pumps
    iscale.set_scaling_factor(m.fs.pump_feed.control_volume.work, 1e-3)
    iscale.set_scaling_factor(m.fs.pump_brine.control_volume.work, 1e-3)
    iscale.set_scaling_factor(m.fs.pump_distillate.control_volume.work, 1e-3)

    # distillate HX
    iscale.set_scaling_factor(m.fs.hx_distillate.hot.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.hx_distillate.cold.heat, 1e-3)
    iscale.set_scaling_factor(
        m.fs.hx_distillate.overall_heat_transfer_coefficient, 1e-3
    )
    iscale.set_scaling_factor(m.fs.hx_distillate.area, 1)
    iscale.constraint_scaling_transform(m.fs.hx_distillate.cold.pressure_drop, 1e-5)
    iscale.constraint_scaling_transform(m.fs.hx_distillate.hot.pressure_drop, 1e-5)

    # evaporator
    iscale.set_scaling_factor(m.fs.evaporator.area, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.U, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_in, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_out, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.lmtd, 1e-1)

    # compressor
    iscale.set_scaling_factor(m.fs.compressor.control_volume.work, 1e-6)

    # condenser
    iscale.set_scaling_factor(m.fs.condenser.control_volume.heat, 1e-6)

    # calculate and propagate scaling factors
    iscale.calculate_scaling_factors(m)

    return m


def add_costing(m):
    m.fs.costing = WaterTAPCosting()
    m.fs.pump_feed.costing = UnitModelCostingBlock(
        default={"flowsheet_costing_block": m.fs.costing}
    )
    m.fs.pump_distillate.costing = UnitModelCostingBlock(
        default={"flowsheet_costing_block": m.fs.costing}
    )
    m.fs.pump_brine.costing = UnitModelCostingBlock(
        default={"flowsheet_costing_block": m.fs.costing}
    )
    m.fs.evaporator.costing = UnitModelCostingBlock(
        default={'flowsheet_costing_block': m.fs.costing}
    )
    m.fs.compressor.costing = UnitModelCostingBlock(
        default={'flowsheet_costing_block': m.fs.costing}
    )

    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.distillate.properties[0].flow_vol)


def add_pressure_drop_to_hx(hx_blk, time_point):
    # input: hx_blk - heat exchanger block
    # output: deactivates control volume pressure balance and adds pressure drop to pressure balance equation
    hx_blk.cold.pressure_balance.deactivate()
    hx_blk.hot.pressure_balance.deactivate()
    hx_blk.cold.deltaP = Var(time_point, initialize=7e4, units=pyunits.Pa)
    hx_blk.hot.deltaP = Var(time_point, initialize=7e4, units=pyunits.Pa)
    hx_blk.cold.pressure_drop = Constraint(
        expr=hx_blk.cold.properties_in[0].pressure
        == hx_blk.cold.properties_out[0].pressure + hx_blk.cold.deltaP[0]
    )
    hx_blk.hot.pressure_drop = Constraint(
        expr=hx_blk.hot.properties_in[0].pressure
        == hx_blk.hot.properties_out[0].pressure + hx_blk.hot.deltaP[0]
    )


def set_operating_conditions(m):
    # Feed inlet
    m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(10)
    m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].fix(0.05)
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)
    m.fs.feed.properties[0].pressure.fix(101325)

    # Feed pump
    m.fs.pump_feed.efficiency_pump.fix(0.8)
    m.fs.pump_feed.control_volume.deltaP[0].fix(7e3)

    # Heat exchangers
    m.fs.hx_distillate.overall_heat_transfer_coefficient.fix(2e3)
    m.fs.hx_distillate.area.fix(50)
    m.fs.hx_distillate.cold.deltaP[0].fix(7e3)
    m.fs.hx_distillate.hot.deltaP[0].fix(7e3)

    # evaporator specifications
    m.fs.evaporator.inlet_feed.temperature[0].fix(273.15 + 45)
    m.fs.evaporator.outlet_brine.temperature[0].fix(273.15 + 60)
    m.fs.evaporator.U.fix(1e3)  # W/K-m^2
    # m.fs.evaporator.area.fix(400)  # m^2
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].fix(5)

    # compressor
    m.fs.compressor.pressure_ratio.fix(2)
    #m.fs.compressor.control_volume.work.fix(5.8521e05)
    m.fs.compressor.efficiency.fix(0.8)

    # Brine pump
    m.fs.pump_brine.efficiency_pump.fix(0.8)
    m.fs.pump_brine.control_volume.deltaP[0].fix(4e4)

    # Distillate pump
    m.fs.pump_distillate.efficiency_pump.fix(0.8)
    m.fs.pump_brine.control_volume.deltaP[0].fix(4e4)

    # Brine outlet
    # m.fs.brine.properties[0].pressure.fix(101325)

    # Distillate outlet
    #m.fs.distillate.properties[0].pressure.fix(101325)

    # Fix 0 TDS
    m.fs.tb_distillate.properties_out[0].flow_mass_phase_comp['Liq','TDS'].fix(1e-5)

    # check degrees of freedom
    print("DOF after setting operating conditions: ", degrees_of_freedom(m))


def initialize_system(m, solver=None):
    if solver is None:
        solver = get_solver()
    optarg = solver.options

    # initialize feed pump
    propagate_state(m.fs.s01)
    m.fs.pump_feed.control_volume.properties_out[0].pressure = (
        m.fs.hx_distillate.cold.deltaP[0].value
        + m.fs.pump_feed.control_volume.properties_in[0].pressure.value
    )
    m.fs.pump_feed.initialize(optarg=optarg)

    # initialize distillate heat exchanger
    propagate_state(m.fs.s02)
    m.fs.hx_distillate.cold_outlet.temperature[
        0
    ] = m.fs.evaporator.inlet_feed.temperature[0].value
    m.fs.hx_distillate.cold_outlet.pressure[0] = m.fs.evaporator.inlet_feed.pressure[
        0
    ].value
    m.fs.hx_distillate.hot_inlet.flow_mass_phase_comp[0, "Liq", "H2O"] = (
        m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].value
    )
    m.fs.hx_distillate.hot_inlet.flow_mass_phase_comp[0, "Liq", "TDS"] = 1e-8
    m.fs.hx_distillate.hot_inlet.temperature[
        0
    ] = m.fs.evaporator.outlet_brine.temperature[0].value
    m.fs.hx_distillate.hot_inlet.pressure[0] = (
        m.fs.distillate.properties[0].pressure.value
        + m.fs.hx_distillate.hot.deltaP[0].value
    )
    m.fs.hx_distillate.initialize()

    # initialize evaporator
    propagate_state(m.fs.s03)
    m.fs.evaporator.initialize_build(
        delta_temperature_in=30, delta_temperature_out=5
    )  # fixes and unfixes those values

    # initialize compressor
    propagate_state(m.fs.s04)
    m.fs.compressor.initialize_build()

    # initialize condenser
    propagate_state(m.fs.s05)
    m.fs.condenser.initialize_build(heat=-m.fs.evaporator.heat_transfer.value)

    # initialize brine pump
    propagate_state(m.fs.s06)
    m.fs.pump_brine.initialize(optarg=optarg)

    # initialize distillate pump
    propagate_state(m.fs.s08)
    propagate_state(m.fs.s09)
    m.fs.pump_distillate.initialize(optarg=optarg)

    # m.fs.costing.initialize()
    print('DOF after initialization: ', degrees_of_freedom(m))


def display_system(m):
    recovery = m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp[
        "Vap", "H2O"
    ].value / (
        m.fs.evaporator.properties_feed[0].flow_mass_phase_comp["Liq", "TDS"].value
        + m.fs.evaporator.properties_feed[0].flow_mass_phase_comp["Liq", "H2O"].value
    )
    print(
        "Feed salinity: ",
        m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].value * 1e3,
        " g/kg",
    )
    print(
        "Brine salinity: ",
        m.fs.brine.properties[0].mass_frac_phase_comp["Liq", "TDS"].value * 1e3,
        " g/kg",
    )
    print("Recovery: ", recovery)
    print("\nDistillate heat exchanger")
    print("Area: ", m.fs.hx_distillate.area.value, " m^2")
    print(
        "U: ", m.fs.hx_distillate.overall_heat_transfer_coefficient[0].value, " W/m^2-K"
    )
    print("Heat transfer: ", m.fs.hx_distillate.heat_duty[0].value, " W")
    print("\nEvaporator")
    print(
        "Temperature: ", m.fs.evaporator.properties_brine[0].temperature.value, " K"
    )  # , ', Fixed? ',  m.fs.evaporator.outlet_brine.temperature[0].fixed())
    print("Pressure: ", m.fs.evaporator.properties_brine[0].pressure.value, " Pa")
    print(
        "Area: ", m.fs.evaporator.area.value, " m^2"
    )  # , ', Fixed? ', m.fs.evaporator.area.isfixed())
    print(
        "U: ", m.fs.evaporator.U.value, " W/m^2-K"
    )  # , ', Fixed? ', m.fs.evaporator.U.isfixed())
    print("heat transfer: ", m.fs.evaporator.heat_transfer.value, " W")
    print("LMTD: ", m.fs.evaporator.lmtd.value, " K")
    print("\nCompressor")
    print("Work: ", m.fs.compressor.control_volume.work[0].value, " W")
    print("Pressure ratio: ", m.fs.compressor.pressure_ratio.value)
    print("Efficiency: ", m.fs.compressor.efficiency.value)
    print("\nCondenser")
    print("Heat transfer: ", m.fs.condenser.control_volume.heat[0].value, " W")
    print("\n Outcome Metrics")
    print('Evaporator capital cost: ', m.fs.evaporator.costing.capital_cost.value)
    print('Compressor capital cost: ', m.fs.compressor.costing.capital_cost.value)
    print(
        "Energy consumption: ",
        value(m.fs.costing.specific_energy_consumption),
        " kWh/m3",
    )
    print("LCOW: ", value(m.fs.costing.LCOW), " $/m3")


def display_seawater_states(state_blk):
    print("water mass flow ", state_blk.flow_mass_phase_comp["Liq", "H2O"].value)
    print("TDS mass flow   ", state_blk.flow_mass_phase_comp["Liq", "TDS"].value)
    print("temperature     ", state_blk.temperature.value)
    print("pressure        ", state_blk.pressure.value)


def display_water_states(state_blk):
    print("Liquid mass flow ", state_blk.flow_mass_phase_comp["Liq", "H2O"].value)
    print("Vapor mass flow  ", state_blk.flow_mass_phase_comp["Vap", "H2O"].value)
    print("temperature      ", state_blk.temperature.value)
    print("pressure         ", state_blk.pressure.value)

    # print("Feed inlet enth_flow: ", value(m.fs.evaporator.properties_feed[0].enth_flow))
    # print(
    #     "Brine inlet enth_flow: ", value(m.fs.evaporator.properties_brine[0].enth_flow)
    # )
    # print(
    #     "Vapor inlet enth_flow: ",
    #     m.fs.evaporator.properties_vapor[0].enth_flow_phase["Vap"].value,


def optimize(m):
    solver = get_solver()
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)
    print("\nSet objective to minimize cost")
    m.fs.evaporator.area.unfix()
    m.fs.evaporator.inlet_feed.temperature[0].unfix()
    m.fs.evaporator.outlet_brine.temperature[0].unfix()
    m.fs.compressor.control_volume.work[0].unfix()
    m.fs.hx_distillate.area.unfix()

    results = solver.solve(m, tee=False)
    assert_optimal_termination(results)


def debug_infeasible(m, solver):
    print("\n---infeasible constraints---")
    infeas.log_infeasible_constraints(m)
    print("\n---infeasible bounds---")
    infeas.log_infeasible_bounds(m)
    print("\n---close to bounds---")
    infeas.log_close_to_bounds(m)
    print("\n---poor scaling---")
    iscale.badly_scaled_var_generator(m)
    # Create Degeneracy Hunter object
    print("\n---degeneracy hunter---")
    dh = DegeneracyHunter(m, solver=solver)
    dh.check_residuals(tol=1e-5)
    # dh.find_candidate_equations(verbose=True,tee=True)
    # dh.check_rank_equality_constraints()

if __name__ == "__main__":
    main()