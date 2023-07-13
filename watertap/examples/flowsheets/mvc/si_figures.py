import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mvc_plotting as mvcplot
from idaes.models.unit_models import Product
from idaes.core import FlowsheetBlock
import idaes.core.util.scaling as iscale
from idaes.core.util.model_statistics import degrees_of_freedom
import watertap.property_models.water_prop_pack as props_w
from pyomo.environ import (
    ConcreteModel,
    value,
    Objective,
    Constraint)
from idaes.core.solvers import get_solver
from iapws import iapws95
from watertap.examples.flowsheets.mvc import mvc_single_stage as mvc_full



def main():
    material_factor_map(get_data=False)
    # sensitivity_vapor_temp()
    # plot_h_fg_vs_P()
    # plot_h_vapor_validation()
    # material_factor_map()

def format_plot(fig):
    plt.legend(frameon=False)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.set_size_inches(3.25, 3.25)



def material_factor_map(get_data=False):
    if get_data:
        wf = [25, 50, 75, 100, 125, 150, 175]
        rr = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        w_min = 0.035  # brine salinity
        w_max = 0.26  # brine salinity
        f_min = 3
        f_max = 9
        slope = (f_max - f_min) / (w_max - w_min)
        brine_salinity = np.empty((len(wf), len(rr)))
        material_factor = np.empty((len(wf), len(rr)))

        for i, S in enumerate(wf):
            print('wf: ', S)
            for j, r in enumerate(rr):
                print('rr: ', r)
                brine_salinity[i][j] = S/(1-r)
                material_factor[i][j] = slope*(brine_salinity[i][j]*1e-3-w_min) + f_min
        b = np.transpose(brine_salinity)
        pd.DataFrame(b).to_csv('C:/Users/carso/Documents/MVC/watertap_results/brine_salinity.csv',
                                  index=False)
        m = np.transpose(material_factor)
        pd.DataFrame(m).to_csv('C:/Users/carso/Documents/MVC/watertap_results/material_factor.csv',
                                  index=False)
    map_dir = 'C:/Users/carso/Documents/MVC/watertap_results/'
    save_dir = map_dir + 'SI_figures'
    var = 'material_factor'
    label = r'Material factor (-)'
    vmin = 3  # minimum cost on bar, $/m3
    vmax = 9  # maximum cost on bar, $/m3
    ticks = [3, 4, 5, 6, 7, 8, 9]  # tick marks on bar
    fmt = '.1f'  # format of annotation
    mvcplot.plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)
    var = 'brine_salinity'
    label = r'Brine salinity (g/kg)'
    vmin = 0  # minimum cost on bar, $/m3
    vmax = 250  # maximum cost on bar, $/m3
    ticks = [0,50, 100, 150, 200, 250]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    mvcplot.plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)


def sensitivity_vapor_temp():
    # Set up
    m = mvc_full.build()
    mvc_full.add_Q_ext(m, time_point=m.fs.config.time)
    mvc_full.set_operating_conditions(m)
    mvc_full.initialize_system(m)
    mvc_full.scale_costs(m)
    mvc_full.fix_outlet_pressures(m)
    m.fs.objective = Objective(expr=m.fs.Q_ext[0])

    solver = get_solver()
    results = mvc_full.solve(m, solver=solver, tee=False)
    mvc_full.add_evap_hx_material_factor_equal_constraint(m)
    mvc_full.add_material_factor_brine_salinity_constraint(m)
    m.fs.Q_ext[0].fix(0)  # no longer want external heating in evaporator
    del m.fs.objective
    mvc_full.set_up_optimization(m)
    results = mvc_full.solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    mvc_full.display_metrics(m)
    mvc_full.display_design(m)

    # solve again with new case
    m.fs.feed.properties[0].mass_frac_phase_comp['Liq', 'TDS'].fix(0.0750)
    m.fs.recovery[0].fix(0.7)
    results = mvc_full.solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    mvc_full.display_metrics(m)
    mvc_full.display_design(m)

    # resolve with change in vapor temp
    m.fs.evaporator.eq_vapor_temperature.deactivate()
    m.fs.evaporator.eq_vapor_temperature_average = Constraint(expr=m.fs.evaporator.properties_vapor[0].temperature == 0.5*(m.fs.evaporator.properties_brine[0].temperature+m.fs.evaporator.properties_feed[0].temperature))
    m.fs.evaporator.properties_brine[0].temperature.setub(75 + 273.15)
    results = mvc_full.solve(m, solver=solver)
    print("Termination condition: ", results.solver.termination_condition)
    mvc_full.display_metrics(m)
    mvc_full.display_design(m)
    print(m.fs.evaporator.properties_vapor[0].temperature.value)

def plot_h_vapor_validation():
    t = np.linspace(100, 175,16) # C
    T = t + 273.15  # K
    P = [25,50,75,101]  # kPa
    # get references
    h = h_sw(273.15+25, 50*1e-3,s=0)
    print(h)
    h_0 = h_sw(273.15+25, P[-1]*1e-3,s=0, has_vap_phase=False)*1e-3
    print(h_0)
    print('Correlation ref: ', h_0)
    state = iapws95.IAPWS95(T=273.15+25, P=P[-1]*1e-3)
    h_0_iapws = state.h # kJ/kg
    print('IAPWS ref: ', h_0_iapws)
    # Set up dict
    h_vapor = {}
    h_liq = {}
    for i,p in enumerate(P):
        print('Pressure: ', p)
        # Get saturation temp
        nist = pd.read_table("C:/Users/carso/Documents/MVC/NIST_water_data/fluid_" + str(p) + 'kPa.txt')
        nist_vapor = nist.loc[nist['Phase'] == 'vapor']
        sat_temp = nist_vapor['Temperature (C)'].iloc[0]
        print('Saturation temp: ', sat_temp)
        t_p = np.linspace(sat_temp + 0.1, 175, 16) # C
        h_vapor['T_'+str(p)] = t_p
        h_vapor['P_'+str(p)+'_cor'] = np.zeros((len(t_p),1))
        h_vapor['P_'+str(p)+'_iapws'] = np.zeros((len(t_p),1))
        t_liq = np.linspace(25,sat_temp-0.1,16) # C
        h_liq['T_'+str(p)] = t_liq
        h_liq['P_' + str(p) + '_cor'] = np.zeros((len(t_liq), 1))
        h_liq['P_' + str(p) + '_iapws'] = np.zeros((len(t_liq), 1))
        for j,temp in enumerate(t_p):
            # h_vapor.append(h_sw(temp,P,s=0, has_vap_phase=True)*1e-3-h_0)
            # h_vap = h_w(temp) + hfg_w(temp)
            h = h_sw(temp+273.15,p*1e-3,s=0,has_vap_phase=True)
            h_vapor['P_' + str(p) + '_cor'][j] = h*1e-3-h_0
            state = iapws95.IAPWS95(T=temp+273.15, P=p*1e-3)
            h_vapor['P_' + str(p) + '_iapws'][j] = state.h-h_0_iapws
        for j, temp in enumerate(t_liq):
            h = h_sw(temp+273.15,p*1e-3,s=0,has_vap_phase=False)
            h_liq['P_' + str(p) + '_cor'][j] = h*1e-3-h_0
            state = iapws95.IAPWS95(T=temp + 273.15, P=p * 1e-3)
            h_liq['P_' + str(p) + '_iapws'][j] = state.h - h_0_iapws

    # Plot
    ls = ['--', ':', '-.', '-']
    # fig = plt.figure
    # for p in P:
    #     plt.plot(h_vapor['T_'+str(p)], h_vapor['P_'+str(p)+'_iapws'],label='IAPWS P='+str(p)+' kPa')
    #     plt.plot(h_vapor['T_'+str(p)], h_vapor['P_'+str(p)+'_cor'],'.',label='Nayar P='+str(p)+' kPa')
    # plt.xlabel('Temperature (C)')
    # plt.ylabel('Specific enthalpy (kJ/kg)')
    # plt.legend()
    # plt.show()

    # Plot error
    fig = plt.figure()
    for i,p in enumerate(P):
        dif = (h_vapor['P_'+str(p)+'_cor']/h_vapor['P_'+str(p)+'_iapws']-1)*100
        plt.plot(h_vapor['T_'+str(p)], dif, 'k',linestyle=ls[i], label='P='+str(p)+' kPa')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Percentage difference in specific enthalpy (%)')
    plt.xlim([65,175])
    plt.ylim([-2,0])
    plt.xticks([65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175])
    plt.legend()
    format_plot(fig)
    plt.show()
    fig.savefig("C:/Users/carso/Documents/MVC/watertap_results/Property comparisons/vapor enthalpy iapws error.svg", bbox_inches='tight', dpi=300)

    # fig = plt.figure
    # for p in P:
    #     plt.plot(h_liq['T_'+str(p)], h_liq['P_'+str(p)+'_iapws'],label='IAPWS P='+str(p)+' kPa')
    #     plt.plot(h_liq['T_'+str(p)], h_liq['P_'+str(p)+'_cor'],'.',label='Nayar P='+str(p)+' kPa')
    # plt.xlabel('Temperature (C)')
    # plt.ylabel('Specific enthalpy (kJ/kg)')
    # plt.legend()
    # plt.show()


def plot_h_fg_vs_P():
    # at saturated conditions
    p_bar = np.linspace(0.1,1,10) # [bar]
    P = p_bar*100 # kPa
    h_fg = [2392.94, 2358.4, 2336.13, 2319.23, 2305.42, 2293.64, 2283.3, 2274.05, 2265.65, 2257.92]
    fig = plt.figure()
    plt.plot(P, h_fg, 'k', label='At saturation T')
    # plt.show()
    # fig.savefig("C:/Users/carso/Documents/MVC/watertap_results/Property comparisons/hfg vs pressure saturated.svg", bbox_inches='tight', dpi=300)

    # # get dh vap vs P
    t = np.array([100, 150]) # C
    T = t + 273.15
    for temp in T:
        h_fg = []
        for p in P:
            state = iapws95.IAPWS95(P=p*1e-3,x=0)
            print(state.h)
            hvap = state.Hvap
            h_fg.append(hvap)
        print(h_fg)
        plt.plot(P,h_fg,label='T = '+str(temp) + ' C')

    plt.xlabel('Pressure (kPa)')
    plt.ylabel('Enthalpy of vaporization (J/kg)')
    format_plot(fig)
    plt.show()


def h_w(T):
    # Input: T - temperature (K)
    # Output: h - specific enthalpy of pure water (J/kg)
    # Validity and accuracy: 5<t<200 C
    # Source: Sharqawy

    t = T-273.15  # [C]

    a0 = 141.355
    a1 = 4202.07
    a2 = -0.535
    a3 = 0.004

    h = (a0 + a1*t + a2*t**2 + a3*t**3)

    return h


def h_sw_sharqawy(T, X=0, has_vap_phase=False):
    # Inputs: T - temperature (K), X - mass fraction of salt (kg/kg)
    # Output: h_sw - specific enthalpy of seawater (kJ/kg)
    # Validity and accuracy: 10<t<120 C and 0<S<120 g/kg

    # enthalpy of pure water
    h = h_w(T)  # [kJ/kg]

    t = T-273.15  # [C]

    a1 = -2.348e4
    a2 = 3.152e5
    a3 = 2.803e6
    a4 = -1.446e7
    a5 = 7.826e3
    a6 = -4.417e1
    a7 = 2.139e-1
    a8 = -1.991e4
    a9 = 2.778e4
    a10 = 9.728e1

    h_calc = h - (X*(a1 + a2*X + a3*X**2 + a4*X**3 + a5*t + a6*t**2 + a7*t**3 + a8*X*t + a9*X**2*t + a10*X*t**2))*1e-3
    # h_calc = h - ((X*(a1 + X) + X*(a2 + X)*t))*1e-3

    if has_vap_phase:
        h_calc = h_calc + hfg_w(T)

    return h_calc


def h_sw(T, P, s=0, has_vap_phase=False):
    # Inputs: T - temperature (K), P - pressure (MPa), X - salinity (g/kg)
    # Output: h - specific enthalpy of seawater (J/kg)
    t = T-273.15 # [C]
    P0 = 0.101 #[MPa]
    X = s*1e-3

    a1 = 9.967767e2
    a2 = -3.2406
    a3 = 0.0127
    a4 = -4.7723e-5
    a5 = -1.1748
    a6 = 0.01169
    a7 = -2.6185e-5
    a8 = 7.0661e-8

    b1 = -2.34825e4
    b2 = 3.15183e5
    b3 = 2.80269e6
    b4 = -1.44606e7
    b5 = 7.82607e3
    b6 = -4.41733e1
    b7 = 2.1394e-1
    b8 = -1.99108e4
    b9 = 2.77846e4
    b10 = 9.72801e1

    hsw_ts0 = h_w(T) - X*(b1 + b2*X + b3*X**2 + b4*X**3 + b5*t + b6*t**2 + b7*t**3 + b8*X*t + b9*X**2*t + b10*X*t**2)   # J/kg

    h_calc = hsw_ts0 + (P-P0)*(a1 + a2*t + a3*t**2 + a4*t**3 + s*(a5 + a6*t + a7*t**2 + a8*t**3))   # J/kg

    if has_vap_phase:
        h_calc = h_calc + hfg_w(T)

    return h_calc


def hfg_w(T):
    # Input: T -  temperature(K)
    # Output: h_fg - latent heat of vaporization of water (J/kg)
    # Source: Equation 55 in Sharqawy (2010)
    # Validity and accuracy: 0 < t, 200 C

    t = T-273.15 # convert to celsius

    a0 = 2.501e6
    a1 = -2.369e3
    a2 = 2.678e-1
    a3 = -8.103e-3
    a4 = -2.079e-5

    h_fg = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 # (J/kg)

    return h_fg


if __name__ == "__main__":
    m = main()