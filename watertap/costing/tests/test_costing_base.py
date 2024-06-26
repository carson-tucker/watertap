#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

import pytest

import pyomo.environ as pyo
import idaes.core as idc

from watertap.costing.watertap_costing_package import WaterTAPCosting


@pytest.mark.component
def test_watertap_costing_package():
    m = pyo.ConcreteModel()
    m.fs = idc.FlowsheetBlock(dynamic=False)

    m.fs.costing = WaterTAPCosting()

    m.fs.electricity = pyo.Var(units=pyo.units.kW, initialize=1)

    m.fs.costing.cost_flow(m.fs.electricity, "electricity")

    assert "foo" not in m.fs.costing.flow_types
    with pytest.raises(
        ValueError,
        match="foo is not a recognized flow type. Please check "
        "your spelling and that the flow type has been registered with"
        " the FlowsheetCostingBlock.",
    ):
        m.fs.costing.cost_flow(m.fs.electricity, "foo")

    m.fs.costing.foo_cost = foo_cost = pyo.Var(
        initialize=42,
        doc="foo",
        units=pyo.units.USD_2020 / pyo.units.m / pyo.units.second,
    )

    m.fs.costing.register_flow_type("foo", m.fs.costing.foo_cost)

    # make sure the component was not replaced
    # by register_defined_flow
    assert foo_cost is m.fs.costing.foo_cost

    m.fs.foo = pyo.Var(units=pyo.units.m, initialize=10)

    m.fs.costing.cost_flow(m.fs.foo, "foo")

    m.fs.costing.bar_base_cost = pyo.Var(
        initialize=0.42,
        doc="bar",
        units=pyo.units.USD_2020 / pyo.units.g / pyo.units.hour,
    )
    m.fs.costing.bar_purity = pyo.Param(
        initialize=0.50, doc="bar purity", units=pyo.units.dimensionless
    )

    m.fs.costing.register_flow_type(
        "bar", m.fs.costing.bar_base_cost * m.fs.costing.bar_purity
    )

    bar_cost = m.fs.costing.bar_cost
    assert isinstance(bar_cost, pyo.Expression)
    assert pyo.value(bar_cost) == 0.21

    m.fs.costing.bar_base_cost.value = 1.5
    assert pyo.value(bar_cost) == 0.75

    m.fs.costing.baz_cost = pyo.Var(initialize=5)

    with pytest.raises(
        RuntimeError,
        match="Component baz_cost already exists on fs.costing but is not 42",
    ):
        m.fs.costing.register_flow_type(
            "baz", 42 * pyo.units.USD_2020 / pyo.units.m**2 / pyo.units.day
        )

    m.fs.costing.flow_types.remove("baz")

    m.fs.costing.register_flow_type(
        "ham", 42 * pyo.units.USD_2021 / pyo.units.kg / pyo.units.minute
    )

    assert isinstance(m.fs.costing.ham_cost, pyo.Var)

    m.fs.costing.cost_process()
    # no error, wacc, plant_lifetime fixed
    m.fs.costing.initialize()

    m.fs.costing.capital_recovery_factor.fix()
    with pytest.raises(
        RuntimeError,
        match="Exactly two of the variables fs.costing.plant_lifetime, "
        "fs.costing.wacc, fs.costing.capital_recovery_factor should be "
        "fixed and the other unfixed.",
    ):
        # error, capital_recovery_factor,  wacc, plant_lifetime all fixed
        m.fs.costing.initialize()
    m.fs.costing.wacc.unfix()

    # no error
    m.fs.costing.initialize()

    m.fs.costing.capital_recovery_factor.unfix()
    with pytest.raises(
        RuntimeError,
        match="Exactly two of the variables fs.costing.plant_lifetime, "
        "fs.costing.wacc, fs.costing.capital_recovery_factor should be "
        "fixed and the other unfixed.",
    ):
        # error, capital_recovery_factor, wacc, unfixed
        m.fs.costing.initialize()

    m.fs.costing.plant_lifetime.unfix()
    with pytest.raises(
        RuntimeError,
        match="Exactly two of the variables fs.costing.plant_lifetime, "
        "fs.costing.wacc, fs.costing.capital_recovery_factor should be "
        "fixed and the other unfixed.",
    ):
        # error, capital_recovery_factor, wacc, and plant_lifetime unfixed
        m.fs.costing.initialize()

    m.fs.costing.wacc.fix()
    m.fs.costing.capital_recovery_factor.fix()
    # no error, wacc, capital_recovery_factor fixed
    m.fs.costing.initialize()
