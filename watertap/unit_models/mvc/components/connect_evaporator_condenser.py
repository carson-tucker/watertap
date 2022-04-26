# Utility function for building MVC flowsheet


# def
### EVAPORATOR CONSTRAINTS ###
    # Temperature difference in
    @self.Constraint(self.flowsheet().time, doc="Temperature difference in")
    def eq_delta_temperature_in(b, t):
        return (
            b.delta_temperature_in
            == b.condenser.control_volume.properties_in[t].temperature
            - b.feed_side.properties_brine[t].temperature
        )

    # Temperature difference out
    @self.Constraint(self.flowsheet().time, doc="Temperature difference out")
    def eq_delta_temperature_out(b, t):
        return (
            b.delta_temperature_out
            == b.condenser.control_volume.properties_out[t].temperature
            - b.feed_side.properties_brine[t].temperature
        )

    # log mean temperature
    @self.Constraint(self.flowsheet().time, doc="Log mean temperature difference")
    def eq_lmtd(b, t):
        dT_in = b.delta_temperature_in
        dT_out = b.delta_temperature_out
        temp_units = pyunits.get_units(dT_in)
        dT_avg = (dT_in + dT_out) / 2
        # external function that ruturns the real root, for the cuberoot of negitive
        # numbers, so it will return without error for positive and negitive dT.
        b.cbrt = ExternalFunction(
            library=functions_lib(), function="cbrt", arg_units=[temp_units**3]
        )
        return b.lmtd == b.cbrt((dT_in * dT_out * dT_avg)) * temp_units

    # Heat transfer between feed side and condenser
    @self.Constraint(self.flowsheet().time, doc="Heat transfer balance")
    def eq_heat_balance(b, t):
        return b.feed_side.heat_transfer == -b.condenser.control_volume.heat[t]

    # Evaporator heat transfer
    @self.Constraint(self.flowsheet().time, doc="Evaporator heat transfer")
    def eq_evaporator_heat(b, t):
        return b.feed_side.heat_transfer == b.U * b.area * b.lmtd