import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

def main():
    # plot_iso_curve("LCOW", dif=True)
    # plot_iso_curve("SEC", dif=True)
    # plot_iso_curve("capex opex ratio", dif=True)
    # assert False
    # Map results
    # map_dir = "C:/Users/carso/Documents/MVC/watertap_results/full_parameter_sweeps_P_out_unfixed/evap_temp_max_75/u_evap_3000_d_2000_b_2000/evap_vary_hx_vary/elec_0.1/cv_temp_max_450/comp_cost_1/"
    # save_dir = map_dir+'figures_min_max_scale'
    # make_maps_min_max_scale(map_dir,save_dir)
    # assert False
    plot_P_b_sensitivity()
    # plot_T_b_sensitivity()
    assert False

    cases = {}

    cases['case_1'] = {}
    cases['case_1']['name'] = 'Case 1'
    cases['case_1']['w_f'] = 0.075
    cases['case_1']['rr'] = 0.5
    cases['case_1']['material factor'] = 6.066667
    cases['case_1']['LCOW'] = 4.77E+00 #4.833906
    cases['case_1']['SEC'] = 22.8 #23.20422
    cases['case_1']['co_ratio'] = 0.672 #0.6666017
    cases['case_1']['color']= '#de4142'
    cases['case_1']['color'] = '#a6bddb'
    cases['case_1']['edgecolor']= '#FFFFFF' # '#661112' #'#93191a'

    cases['case_2'] = {}
    cases['case_2']['name'] = 'Case 2'
    cases['case_2']['w_f'] = 0.1
    cases['case_2']['rr'] = 0.5
    cases['case_2']['material factor'] = 7.4
    cases['case_2']['LCOW'] = 5.32 #5.381881
    cases['case_2']['SEC'] = 26.2 #26.71866
    cases['case_2']['co_ratio'] = 0.638 #0.6322312
    cases['case_2']['color']='#0876b9'
    cases['case_2']['color'] = '#3690c0'
    cases['case_2']['edgecolor']= '#FFFFFF' #'#03314d' #'#05476f'

    cases['case_3'] = {}
    cases['case_3']['name'] = 'Case 3'
    cases['case_3']['w_f'] = 0.75
    cases['case_3']['rr'] = 0.7
    cases['case_3']['material factor'] = 8.733333
    cases['case_3']['LCOW'] = 5.69 ## 5.764883
    cases['case_3']['SEC'] = 28.98 #29.79601
    cases['case_3']['co_ratio'] = 0.61 #0.5914722
    cases['case_3']['color']= '#0c9856'
    cases['case_3']['color'] = '#016450'
    cases['case_3']['edgecolor']= '#FFFFFF'#043f24' #'#075b34'


    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/split_ratio_sensitivity_fixed_temp/"
    plot_split_ratio_sensitivity(map_dir, cases)
    assert False

    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/T_f_sensitivity/"
    plot_T_f_sensitivity(map_dir, cases)
    assert False
    # results = map_dir + 'T_b.csv'
    # x_param = "# T_b"
    # y_param = 'LCOW'
    # x_label = 'Temperature (K)'
    # y_label = 'LCOW $/m3'
    # plot_single_param_sensitivity(map_dir, results, x_param, y_param, x_label, y_label)
    # assert False


    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/material_factor_sensitivity/"
    plot_material_factor_sensitivity(map_dir,cases)
    assert False
    # run_dual_param_sensitivity()
    # map_dir = "C:/Users/carso/Documents/MVC/watertap_results/material_factor_sensitivity/"
    # map_dir = "C:/Users/carso/Documents/MVC/watertap_results/tornado_sensitivity/"
    # assert False
    # map_dir = "C:/Users/carso/Documents/MVC/watertap_results/T_b_sensitivity_vary_material_factor/"
    # plot_T_b_sensitivity(map_dir,cases)
    # assert False

    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/tornado_sensitivity/case_1/"
    # save_tornado_plot_data(map_dir, LCOW_base=cases['case_1']['LCOW'], SEC_base=cases['case_1']['SEC'], co_ratio_base=cases['case_1']['co_ratio'])
    # run_tornado(map_dir)
    # assert False
    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/tornado_sensitivity/case_2/"
    # run_tornado(map_dir)
    # save_tornado_plot_data(map_dir, LCOW_base=cases['case_2']['LCOW'], SEC_base=cases['case_2']['SEC'], co_ratio_base=cases['case_2']['co_ratio'])
    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/tornado_sensitivity/case_3/"
    # save_tornado_plot_data(map_dir, LCOW_base=cases['case_3']['LCOW'], SEC_base=cases['case_3']['SEC'], co_ratio_base=cases['case_3']['co_ratio'])
    # run_tornado(map_dir)
    # assert False

    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/tornado_sensitivity/"
    tornado_plot_multiple(map_dir, cases)
    assert False

    # map_dir = "C:/Users/carso/Documents/MVC/watertap_results/T_b_sensitivity_vary_material_factor/Case wf 25 rr 40/"
    # results = map_dir + 'T_b.csv'
    # x_param = "# T_b"
    # y_param = 'LCOW'
    # x_label = 'Temperature (K)'
    # y_label = 'LCOW $/m3'
    # plot_single_param_sensitivity(map_dir, results, x_param, y_param, x_label, y_label)
    # assert False

    return

def plot_iso_curve(param, dif=True):
    # plots dual parameter sensitivity with isocost/isoenergy curve
    fig = plt.figure()
    ax = plt.axes()
    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/dual_F_m_U_evap_sensitivity"
    save_dir = map_dir + "/figures"
    xlabel = "Evaporator cost (%)"
    xticklabels = ['-25', '-20', '-15', '-10', '-5','0','+5','+10', '+15', '+20', '+25']
    ylabel = "Evaporator overall heat transfer\ncoefficient change (%)"
    yticklabels = xticklabels

    # get percentage difference
    results_file = map_dir + '/' + param + '.csv'
    map_data = pd.read_csv(results_file)
    if dif:
        center_x = xticklabels.index("0")
        center_y = yticklabels.index("0")
        base_case = map_data[str(center_x)][center_y]
        map_data = map_data.divide(base_case)-1
        map_data = map_data*100
        label = param + " change (%)"
        param = param + ' percentage difference'
        # cmap = 'coolwarm'
        cmap = 'viridis'
    label = 'CAPEX/OPEX change (%)'
    vmin = -25
    vmax = 25
    ticks = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    fmt = '.0f'

    # Plot contours
    # x = np.array(range(map_data.shape[1]))
    # y = np.array(range(map_data.shape[0]))
    x = ticks
    y = ticks
    xx, yy = np.array(np.meshgrid(x,y))
    z_range = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    cs = ax.contourf(xx, yy, map_data, z_range, cmap=cmap)

    # Add color bar
    fig.colorbar(cs,ax=ax,ticks=z_range,label=label)

    # Plot iso-curve
    levels = [0]
    cs = ax.contour(xx, yy, map_data, levels, colors="black")

    # axis labels
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7, rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.set_size_inches(3.25, 3.25)
    # plt.show()
    fig.savefig(save_dir + '/' + param + '.png', bbox_inches='tight', dpi=300)
    fig.savefig(save_dir + '/' + param + '.svg', bbox_inches='tight', dpi=300)


def plot_least_work():
    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/"
    save_dir = map_dir
    var = 'W_least_est_mistry_kJ_kg'
    label = r'$W_{least}/m_{perm}$ (J/kg)'
    vmin = 0
    vmax = 25
    ticks = [0, 5, 10, 15, 20, 25]
    fmt = '.1f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'W_least_g_direct_nayar_kJ_kg'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)
    #
    var = 'W_least_g_calc_nayar_kJ_kg'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'W_least_kJ_kg'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

def plot_efficiency(map_dir):
    save_dir = map_dir + "figures/"
    var = 'Second law efficiency direct nayar'
    label = r'$\eta_{II}$ (%)'
    vmin = 0
    vmax = 100
    ticks = [0, 20, 40, 60, 80, 100]
    fmt = '.1f'
    save_dir = map_dir + 'figures/'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Second law efficiency calc nayar'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Second law efficiency est mistry'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

def run_tornado(map_dir):
    # map_dir = "C:/Users/carso/Documents/MVC/watertap_results/tornado_sensitivity/case_1/"
    save_dir = map_dir + "figures"
    # save_tornado_plot_data(map_dir)
    tornado_plot(map_dir, save_dir)

def run_tornado_multiple_cases(cases):
    tornado_plot_multiple(map_dir,save_dir,cases)

def run_dual_param_sensitivity():
    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/dual_F_m_U_evap_sensitivity"
    save_dir = map_dir + "/figures"
    xlabel = "Material factor (%)"
    xticklabels = ['-25', '-20', '-15', '-10', '-5','0','+5','+10', '+15', '+20', '+25']
    ylabel = "Evaporator overall heat transfer\ncoefficient change (%)"
    yticklabels = xticklabels
    make_maps_dual_param(map_dir, save_dir, xticklabels, yticklabels, xlabel, ylabel)

def plot_2D_heat_map_dual_param(map_dir, save_dir, param, label, vmin, vmax, ticks, fmt, xticklabels, yticklabels, xlabel, ylabel, make_ticks=True, dif=False):
    fig = plt.figure()
    ax = plt.axes()
    results_file = map_dir + '/' + param + '.csv'
    df = pd.read_csv(results_file)
    cmap='YlGnBu'
    if dif == True:
        center_x = xticklabels.index("0")
        center_y = yticklabels.index("0")
        df_0 = df[str(center_x)][center_y]
        df = df.divide(df_0)-1
        df = df*100
        label = param + " change (%)"
        param = param + ' percentage difference'
        cmap = 'viridis'
        # cmap = 'coolwarm'

    if make_ticks:
        decimal = int(fmt[1])
        df_min = float(np.nanmin(df.values))
        vmin = df_min - 10 ** -decimal
        df_max = float(np.nanmax(df.values))
        vmax = df_max + 10 ** -decimal
        n = 5  # number of ticks
        ticks = np.round(np.linspace(df_min, df_max, n), decimal)  # round based on formatting decimal places

    ax = sns.heatmap(df, cmap=cmap,
                     vmin=vmin, vmax=vmax, annot=True, annot_kws={"fontsize": 7}, fmt=fmt,
                     cbar_kws={'label': label, "ticks": ticks}, xticklabels=xticklabels,
                     yticklabels=yticklabels
                     )  # create heatmap
    ax.invert_yaxis()
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7,rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.set_size_inches(3.25, 3.25)
    # plt.show()
    fig.savefig(save_dir + '/' + param + '.png', bbox_inches='tight', dpi=300)
    fig.savefig(save_dir + '/' + param + '.svg', bbox_inches='tight', dpi=300)


def plot_2D_heat_map(map_dir, save_dir, param, label, vmin, vmax, ticks, fmt, make_ticks=True):
    fig = plt.figure()
    ax = plt.axes()
    results_file = map_dir + '/' + param + '.csv'
    df = pd.read_csv(results_file)
    # print(df)
    df = df.drop(df.columns[6],axis=1) # get rid of 175 g/kg case
    # print(df)
    # mask = df.isnull()
    df_wb = pd.read_csv("C:/Users/carso/Documents/MVC/watertap_results/Brine salinity.csv")
    df_wb = df_wb.drop(df_wb.columns[6],axis=1)
    mask = df_wb > 0.26
    # print(mask.shape)
    # print(df.shape)
    # assert False
    # mask['0'][0] = True
    # mask['0'][1] = True
    if make_ticks:
        decimal = int(fmt[1])
        df_min = float(np.nanmin(df[~mask].values))
        vmin = df_min - 10 ** -decimal
        df_max = float(np.nanmax(df[~mask].values))
        vmax = df_max + 10 ** -decimal
        n = 5  # number of ticks
        ticks = np.round(np.linspace(df_min, df_max, n), decimal)  # round based on formatting decimal places

    xticklabels = ['25', '50', '75', '100', '125', '150']
    # xticklabels = ['10','15','20','25', '30','35','40','45']

    yticklabels = ['40', '45', '50', '55', '60', '65', '70', '75', '80']
    # ax = sns.heatmap(df, cmap='Reds', mask=mask,
    #                  vmin=vmin, vmax=vmax, annot=True, annot_kws={"fontsize": 8}, fmt=fmt,
    #                  cbar_kws={'label': label, "ticks": ticks}, xticklabels=xticklabels,
    #                  yticklabels=yticklabels
    #                  )  # create heatmap
    ax = sns.heatmap(df, cmap='YlGnBu', mask=mask,
                     vmin=vmin, vmax=vmax, annot=True, annot_kws={"fontsize": 8}, fmt=fmt,
                     cbar_kws={'label': label, "ticks": ticks}, xticklabels=xticklabels,
                     yticklabels=yticklabels
                     )  # create heatmap
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    ax.set_xlabel('Feed concentration (g/kg)')
    ax.set_ylabel('Water recovery (%)')
    fig.set_size_inches(3.25, 3.25)
    # plt.show()
    fig.savefig(save_dir + '/' + param + '.png', bbox_inches='tight', dpi=300)
    fig.savefig(save_dir + '/' + param + '.svg', bbox_inches='tight', dpi=300)

def plot_single_param_sensitivity(map_dir, results, x_param, y_param, x_label, y_label):
    # results = map_dir + "/optimize_sweep.csv"
    df = pd.read_csv(results)
    x = df[x_param]-273.15
    y = df[y_param]
    fig = plt.figure()
    ax = plt.axes()
    plt.plot(x,y,'r')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlabel(x_label,fontsize=12)
    ax.set_ylabel(y_label,fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/' + y_param + ' vs ' + x_param + '.png', bbox_inches='tight', dpi=300)

def plot_compressor_efficiency_sensitivity(map_dir,cases):
    # results = map_dir + "/optimize_sweep.csv"
    fig = plt.figure()
    ax = plt.axes()
    for name,case in cases.items():
        results = map_dir + case['name'] +'/compressor_efficiency.csv'
        df = pd.read_csv(results)
        x = df['# compressor_efficiency']
        y = df['LCOW']
        plt.plot(x,y,color=case['color'],label=case['name'])
    for name,case in cases.items():
        plt.scatter(0.8, case['LCOW'],marker="*", facecolors=case['color'], edgecolors=case['color'])
    plt.xticks([0.7,0.75,0.8,0.85,0.9], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0,7)
    plt.xlim(0.7,0.9)
    plt.legend(frameon=False)
    ax.set_xlabel('Compressor Efficiency (-)',fontsize=12)
    ax.set_ylabel(r'LCOW (\$/$\rmm^3$ of product)',fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/LCOW vs compressor efficiency.png', bbox_inches='tight', dpi=300)
    # SEC
    fig = plt.figure()
    ax = plt.axes()
    for name,case in cases.items():
        results = map_dir + case['name'] +'/compressor_efficiency.csv'
        df = pd.read_csv(results)
        x = df['# compressor_efficiency']
        y = df['SEC']
        plt.plot(x,y,color=case['color'],label=case['name'])
    for name,case in cases.items():
        plt.scatter(0.8, case['SEC'],marker="*", facecolors=case['color'], edgecolors=case['color'])
    plt.xticks([0.7,0.75,0.8,0.85,0.9], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(20,35)
    plt.xlim(0.7,0.9)
    plt.legend(frameon=False)
    ax.set_xlabel('Compressor Efficiency (-)',fontsize=12)
    ax.set_ylabel(r'SEC (kWh/$\rmm^3$ of product)',fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/SEC vs compressor efficiency.png', bbox_inches='tight', dpi=300)

    # CO Ratio
    fig = plt.figure()
    ax = plt.axes()
    for name,case in cases.items():
        results = map_dir + case['name'] +'/compressor_efficiency.csv'
        df = pd.read_csv(results)
        x = df['# compressor_efficiency']
        y = df['capex opex ratio']
        plt.plot(x,y,color=case['color'],label=case['name'])
    for name,case in cases.items():
        plt.scatter(0.8, case['co_ratio'],marker="*", facecolors=case['color'], edgecolors=case['color'])
    plt.xticks([0.7,0.75,0.8,0.85,0.9], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0,1)
    plt.xlim(0.7,0.9)
    plt.legend(frameon=False)
    ax.set_xlabel('Compressor Efficiency (-)',fontsize=12)
    ax.set_ylabel('CAPEX/OPEX ratio (-)',fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/capex opex ratio vs compressor efficiency.png', bbox_inches='tight', dpi=300)

def plot_pressure_ratio_sensitivity():
    fig = plt.figure()
    ax = plt.axes()
    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/pressure_ratio_sensitivity"
    results = "C:/Users/carso/Documents/MVC/watertap_results/pressure_ratio_sensitivity/pressure_ratio.csv"
    df = pd.read_csv(results)
    x = df['pressure_ratio']
    y = df['LCOW']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(1.76, 5.32, marker="*", facecolors='#3690c0',
                    edgecolors='#3690c0')
    # plt.xticks([4,5,6,7,8,9, 10,11,12], fontsize=12)
    plt.xticks([1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 10)
    # plt.xlim(4,12)
    plt.xlim(1.3,2)
    plt.legend(frameon=False)
    ax.set_xlabel('Compressor pressure ratio (-)', fontsize=12)
    ax.set_ylabel(r'LCOW (\$/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/LCOW vs pr.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/LCOW vs pr.svg', bbox_inches='tight', dpi=300)

    # SEC
    fig = plt.figure()
    ax = plt.axes()
    y = df['SEC']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(1.76, 26.2 , marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    # plt.xticks([4,5,6,7,8,9, 10,11,12], fontsize=12)
    plt.xticks([1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 35)
    # plt.xlim(4,12)
    plt.xlim(1.3, 2)
    plt.legend(frameon=False)
    ax.set_xlabel('Compressor pressure ratio (-)', fontsize=12)
    ax.set_ylabel(r'SEC (kWh/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/SEC vs pr.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/SEC vs pr.svg', bbox_inches='tight', dpi=300)

    # evaporator area
    fig = plt.figure()
    ax = plt.axes()
    y = df['Evaporator area']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(1.76, 631, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    plt.xticks([1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 2000)
    plt.xlim(1.3, 2)
    plt.legend(frameon=False)
    ax.set_xlabel('Compressor pressure ratio (-)', fontsize=12)
    ax.set_ylabel(r'Evaporator area ($m^2$)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/evap area vs pr.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/evap area vs pr.svg', bbox_inches='tight', dpi=300)

def plot_split_ratio_sensitivity(map_dir,cases):
    fig = plt.figure()
    ax = plt.axes()
    for name, case in cases.items():
        print(case)
        # results = map_dir + case['name'] +'/material_factor_25p.csv'
        results = map_dir + case['name'] + '/results.csv'
        df = pd.read_csv(results)
        x = df['# split_ratio']
        y = df['LCOW']
        plt.plot(x, y, color=case['color'], label=case['name'])
    for name, case in cases.items():
        plt.scatter(case['rr'], case['LCOW'], marker="*", facecolors=case['color'],
                    edgecolors=case['color'])
    # plt.xticks([4,5,6,7,8,9, 10,11,12], fontsize=12)
    plt.xticks([0.3, 0.4,0.5, 0.6, 0.7,0.8,0.9], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 10)
    # plt.xlim(4,12)
    plt.xlim(0.3,0.9)
    plt.legend(frameon=False)
    ax.set_xlabel('Preheater feed split ratio (-)', fontsize=12)
    ax.set_ylabel(r'LCOW (\$/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/LCOW vs split ratio.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/LCOW vs split ratio.svg', bbox_inches='tight', dpi=300)

    # SEC
    fig = plt.figure()
    ax = plt.axes()
    for name, case in cases.items():
        print(case)
        # results = map_dir + case['name'] + '/material_factor_25p.csv'
        results = map_dir + case['name'] + '/results.csv'
        df = pd.read_csv(results)
        x = df['# split_ratio']
        y = df['SEC']
        plt.plot(x, y, color=case['color'], label=case['name'])
    for name, case in cases.items():
        plt.scatter(case['rr'], case['SEC'], marker="*", facecolors=case['color'],
                    edgecolors=case['color'])
    plt.xticks([0.3, 0.4,0.5, 0.6, 0.7,0.8,0.9], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 40)
    # plt.xlim(4, 12)
    plt.xlim(0.3, 0.9)
    plt.legend(frameon=False)
    ax.set_xlabel('Preheater feed split ratio (-)', fontsize=12)
    ax.set_ylabel(r'SEC (kWh/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/SEC vs split ratio.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/SEC vs split ratio.svg', bbox_inches='tight', dpi=300)

    fig = plt.figure()
    ax = plt.axes()
    for name, case in cases.items():
        # results = map_dir + case['name'] + '/material_factor_25p.csv'
        results = map_dir + case['name'] + '/results.csv'
        df = pd.read_csv(results)
        x = df['# split_ratio']
        y = df['capex opex ratio']
        plt.plot(x, y, color=case['color'], label=case['name'])
    for name, case in cases.items():
        plt.scatter(case['rr'], case['co_ratio'], marker="*", facecolors=case['color'],
                    edgecolors=case['color'])
    plt.xticks([0.3, 0.4,0.5, 0.6, 0.7,0.8,0.9], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.35, 1)
    # plt.xlim(4, 11)
    plt.xlim(0.3, 0.9)
    plt.legend(frameon=False)
    ax.set_xlabel('Preheater feed split ratio (-)', fontsize=12)
    ax.set_ylabel('CAPEX/OPEX Ratio (-)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/capex opex ratio vs split ratio.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/capex opex ratio vs split ratio.svg', bbox_inches='tight', dpi=300)

def plot_material_factor_sensitivity(map_dir,cases):
    # results = map_dir + "/optimize_sweep.csv"
    fig = plt.figure()
    ax = plt.axes()
    for name,case in cases.items():
        print(case)
        # results = map_dir + case['name'] +'/material_factor_25p.csv'
        results = map_dir + case['name'] +'/material_factor.csv'
        df = pd.read_csv(results)
        x = df['# material_factor']
        y = df['LCOW']
        plt.plot(x,y,color=case['color'],label=case['name'])
    for name,case in cases.items():
        plt.scatter(case['material factor'], case['LCOW'],marker="*", facecolors=case['color'], edgecolors=case['color'])
    # plt.xticks([4,5,6,7,8,9, 10,11,12], fontsize=12)
    plt.xticks([3,4,5,6,7,8,9], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0,7)
    # plt.xlim(4,12)
    plt.xlim(3,9)
    plt.legend(frameon=False)
    ax.set_xlabel('Material Factor (-)',fontsize=12)
    ax.set_ylabel(r'LCOW (\$/$m^3$ of product)',fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/LCOW vs material factor.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/LCOW vs material factor.svg', bbox_inches='tight', dpi=300)

    # SEC
    fig = plt.figure()
    ax = plt.axes()
    for name, case in cases.items():
        print(case)
        # results = map_dir + case['name'] + '/material_factor_25p.csv'
        results = map_dir + case['name'] + '/material_factor.csv'
        df = pd.read_csv(results)
        x = df['# material_factor']
        y = df['SEC']
        plt.plot(x, y, color=case['color'], label=case['name'])
    for name, case in cases.items():
        plt.scatter(case['material factor'], case['SEC'], marker="*", facecolors=case['color'],
                    edgecolors=case['color'])
    # plt.xticks([4, 5, 6, 7, 8, 9,10,11,12], fontsize=12)
    plt.xticks([3,4,5,6,7,8,9], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(15, 30)
    # plt.xlim(4, 12)
    plt.xlim(3, 9)
    plt.legend(frameon=False)
    ax.set_xlabel('Material Factor (-)', fontsize=12)
    ax.set_ylabel(r'SEC (kWh/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/SEC vs material factor.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/SEC vs material factor.svg', bbox_inches='tight', dpi=300)

    fig = plt.figure()
    ax = plt.axes()
    for name, case in cases.items():
        # results = map_dir + case['name'] + '/material_factor_25p.csv'
        results = map_dir + case['name'] + '/material_factor.csv'
        df = pd.read_csv(results)
        x = df['# material_factor']
        y = df['capex opex ratio']
        plt.plot(x, y, color=case['color'], label=case['name'])
    for name, case in cases.items():
        plt.scatter(case['material factor'], case['co_ratio'], marker="*", facecolors=case['color'],
                    edgecolors=case['color'])
    # plt.xticks([4, 5, 6, 7, 8, 9,10,11,12], fontsize=12)
    plt.xticks([3,4,5,6,7,8,9], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.5, 0.7)
    # plt.xlim(4, 11)
    plt.xlim(3, 9)
    plt.legend(frameon=False)
    ax.set_xlabel('Material Factor (-)', fontsize=12)
    ax.set_ylabel('CAPEX/OPEX Ratio (-)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/capex opex ratio vs material factor.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/capex opex ratio vs material factor.svg', bbox_inches='tight', dpi=300)


def plot_T_f_sensitivity(map_dir,cases):
    # results = map_dir + "/optimize_sweep.csv"
    fig = plt.figure()
    ax = plt.axes()
    for name,case in cases.items():
        print(case)
        results = map_dir + case['name'] +'/T_f.csv'
        df = pd.read_csv(results)
        x = df['T_f']
        y = df['LCOW']
        plt.plot(x,y,color=case['color'],label=case['name'])
    # for name,case in cases.items():
    #     plt.scatter(25, case['LCOW'],marker="*", facecolors=case['color'], edgecolors=case['color'])
    # plt.xticks([4,5,6,7,8,9, 10,11,12], fontsize=12)
    plt.xticks([25,30,35,40,45], fontsize=12)
    plt.xticks([55, 65, 75, 85, 95], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0,7)
    # plt.xlim(4,12)
    plt.xlim(55,95)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator Temperature (C)',fontsize=12)
    ax.set_ylabel(r'LCOW (\$/$m^3$ of product)',fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    # fig.savefig(map_dir + '/LCOW vs feed temperature.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/LCOW vs feed temperature EVAPORATOR LABELS.svg', bbox_inches='tight', dpi=300)

    # SEC
    fig = plt.figure()
    ax = plt.axes()
    for name, case in cases.items():
        print(case)
        results = map_dir + case['name'] +'/T_f.csv'
        df = pd.read_csv(results)
        x = df['T_f']
        y = df['SEC']
        plt.plot(x, y, color=case['color'], label=case['name'])
    # for name, case in cases.items():
    #     plt.scatter(25, case['SEC'], marker="*", facecolors=case['color'],
    #                 edgecolors=case['color'])
    plt.xticks([25,30,35,40,45], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(15, 30)
    # plt.xlim(4, 12)
    plt.xlim(25, 45)
    plt.legend(frameon=False)
    ax.set_xlabel('Feed Temperature (C)',fontsize=12)
    ax.set_ylabel(r'SEC (kWh/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/SEC vs feed temperature.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/SEC vs feed temperature.svg', bbox_inches='tight', dpi=300)

    # Evaporator area
    fig = plt.figure()
    ax = plt.axes()
    for name, case in cases.items():
        results = map_dir + case['name'] + '/T_f.csv'
        df = pd.read_csv(results)
        x = df['T_f']
        y = df['Evaporator area']
        plt.plot(x, y, color=case['color'], label=case['name'])
    # for name, case in cases.items():
    #     plt.scatter(case['material factor'], case['co_ratio'], marker="*", facecolors=case['color'],
    #                 edgecolors=case['color'])
    plt.xticks([25,30,35,40,45], fontsize=12)
    plt.yticks(fontsize=12)
    # plt.ylim(0.5, 0.7)
    # plt.xlim(4, 11)
    plt.xlim(25, 45)
    plt.legend(frameon=False)
    ax.set_xlabel('Feed Temperature (C)',fontsize=12)
    ax.set_ylabel(r'Evaporator Area ($m^2$)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/evaporator area vs feed temperature.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/evaporator area vs feed temperature.svg', bbox_inches='tight', dpi=300)


    # Total preheater area
    fig = plt.figure()
    ax = plt.axes()
    for name, case in cases.items():
        results = map_dir + case['name'] + '/T_f.csv'
        df = pd.read_csv(results)
        x = df['T_f']
        y = df['Distillate hx area'] + df['Brine hx area']
        plt.plot(x, y, color=case['color'], label=case['name'])
    # for name, case in cases.items():
    #     plt.scatter(case['material factor'], case['co_ratio'], marker="*", facecolors=case['color'],
    #                 edgecolors=case['color'])
    plt.xticks([25,30,35,40,45], fontsize=12)
    plt.yticks(fontsize=12)
    # plt.ylim(0.5, 0.7)
    plt.xlim(25, 45)
    plt.legend(frameon=False)
    ax.set_xlabel('Feed Temperature (C)',fontsize=12)
    ax.set_ylabel(r'Total Preheater Area ($m^2$)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/total preheater area vs feed temperature.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/total preheater area vs feed temperature.svg', bbox_inches='tight', dpi=300)


def plot_T_b_sensitivity():
    fig = plt.figure()
    ax = plt.axes()
    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/T_b_sensitivity_vary_material_factor/Case 2"
    results = "C:/Users/carso/Documents/MVC/watertap_results/T_b_sensitivity_vary_material_factor/Case 2/T_b.csv"
    df = pd.read_csv(results)
    x = df['T_b_celsius']
    y = df['LCOW']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(75, 5.32, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    # plt.xticks([4,5,6,7,8,9, 10,11,12], fontsize=12)
    plt.xticks([55, 65, 75, 85, 95], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 6)
    plt.xlim(55,95)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator temperature (C)', fontsize=12)
    ax.set_ylabel(r'LCOW (\$/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/LCOW vs T_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/LCOW vs T_b.svg', bbox_inches='tight', dpi=300)

    # Evaporator pressure
    fig = plt.figure()
    ax = plt.axes()
    y = df['Brine pressure kPa']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(75, 32.45, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    plt.xticks([55, 65, 75, 85, 95], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 80)
    plt.xlim(55, 95)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator temperature (C)', fontsize=12)
    ax.set_ylabel(r'Evaporator pressure (kPa)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/Evaporator pressure vs T_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/Evaporator pressure vs T_b.svg', bbox_inches='tight', dpi=300)


     # SEC
    fig = plt.figure()
    ax = plt.axes()
    y = df['SEC']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(75, 26.2, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    plt.xticks([55, 65, 75, 85, 95], fontsize=12)
    plt.xlim(55, 95)
    plt.yticks(fontsize=12)
    plt.ylim(0, 35)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator temperature (C)', fontsize=12)
    ax.set_ylabel(r'SEC (kWh/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/SEC vs T_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/SEC vs T_b.svg', bbox_inches='tight', dpi=300)

    # evaporator area
    fig = plt.figure()
    ax = plt.axes()
    y = df['Evaporator area']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(75, 631, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    plt.xticks([55, 65, 75, 85, 95], fontsize=12)
    plt.xlim(55, 95)
    plt.yticks(fontsize=12)
    plt.ylim(0, 2000)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator temperature (C)', fontsize=12)
    ax.set_ylabel(r'Evaporator area ($m^2$)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/evap area vs T_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/evap area vs T_b.svg', bbox_inches='tight', dpi=300)

def plot_P_b_sensitivity():
    fig = plt.figure()
    ax = plt.axes()
    map_dir = "C:/Users/carso/Documents/MVC/watertap_results/T_b_sensitivity_vary_material_factor/Case 2"
    results = "C:/Users/carso/Documents/MVC/watertap_results/T_b_sensitivity_vary_material_factor/Case 2/T_b.csv"
    df = pd.read_csv(results)
    x = df['Brine pressure kPa']
    y = df['LCOW']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(32.45, 5.32, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    plt.ylim(0,7)
    plt.yticks(fontsize=12)
    plt.xlim(13,71.5)
    plt.xticks([20, 30, 40,50,60, 70], fontsize=12)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator pressure (kPa)', fontsize=12)
    ax.set_ylabel(r'LCOW (\$/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/LCOW vs P_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/LCOW vs P_b.svg', bbox_inches='tight', dpi=300)

    # Evaporator temperature
    fig = plt.figure()
    ax = plt.axes()
    y = df['T_b_celsius']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(32.45, 75, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    plt.yticks([55, 65, 75, 85, 95], fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlim(13,71.5)
    plt.xticks([20, 30, 40,50,60, 70], fontsize=12)
    plt.ylim(55, 95)
    plt.legend(frameon=False)
    ax.set_ylabel('Evaporator temperature (C)', fontsize=12)
    ax.set_xlabel(r'Evaporator pressure (kPa)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/Evaporator temp vs P_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/Evaporator temp vs P_b.svg', bbox_inches='tight', dpi=300)


     # SEC
    fig = plt.figure()
    ax = plt.axes()
    y = df['SEC']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(32.45, 26.2, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    # plt.xticks([55, 65, 75, 85, 95], fontsize=12)
    # plt.xlim(55, 95)
    plt.yticks(fontsize=12)
    plt.ylim(0, 35)
    plt.xlim(13,71.5)
    plt.xticks([20, 30, 40,50,60, 70], fontsize=12)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator pressure (kPa)', fontsize=12)
    ax.set_ylabel(r'SEC (kWh/$m^3$ of product)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/SEC vs P_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/SEC vs P_b.svg', bbox_inches='tight', dpi=300)

    # evaporator area
    fig = plt.figure()
    ax = plt.axes()
    y = df['Evaporator area']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(32.45, 631, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')
    # plt.xticks([55, 65, 75, 85, 95], fontsize=12)
    # plt.xlim(55, 95)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1000)
    plt.xlim(13,71.5)
    plt.xticks([20, 30, 40,50,60, 70], fontsize=12)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator pressure (kPa)', fontsize=12)
    ax.set_ylabel(r'Evaporator area ($m^2$)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/evap area vs P_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/evap area vs P_b.svg', bbox_inches='tight', dpi=300)

    # Compressor cost
    fig = plt.figure()
    ax = plt.axes()
    y = df['Compressor capital cost M']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(32.45, 1.154086, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')

    plt.xlim(13,71.5)
    plt.xticks([20, 30, 40,50,60, 70], fontsize=12)
    plt.ylim(1.152, 1.168)
    plt.yticks([1.155, 1.160, 1.165])
    plt.yticks(fontsize=12)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator pressure (kPa)', fontsize=12)
    ax.set_ylabel('Compressor capital cost ($M)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/compressor cost vs P_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/compressor cost vs P_b.svg', bbox_inches='tight', dpi=300)


    # Evaporator cost
    fig = plt.figure()
    ax = plt.axes()
    y = df['Evaporator capital cost M']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(32.45, 4.78163, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')

    plt.xlim(13,71.5)
    plt.xticks([20, 30, 40,50,60, 70], fontsize=12)
    # plt.ylim(1.152, 1.168)
    # plt.yticks([1.155, 1.160, 1.165])
    plt.yticks(fontsize=12)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator pressure (kPa)', fontsize=12)
    ax.set_ylabel('Evaporator capital cost ($M)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/evaporator cost vs P_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/evaporator cost vs P_b.svg', bbox_inches='tight', dpi=300)


    # Q evap
    fig = plt.figure()
    ax = plt.axes()
    y = df['Q_evap normalized']
    plt.plot(x, y, color='#3690c0', label='Case 2')
    plt.scatter(32.45, 2.3736152, marker="*", facecolors='#3690c0',
                edgecolors='#3690c0')

    plt.xlim(13,71.5)
    plt.xticks([20, 30, 40,50,60, 70], fontsize=12)
    # plt.ylim(1.152, 1.168)
    # plt.yticks([1.155, 1.160, 1.165])
    plt.yticks(fontsize=12)
    plt.legend(frameon=False)
    ax.set_xlabel('Evaporator pressure (kPa)', fontsize=12)
    ax.set_ylabel('Evaporator heat transfer\n(MJ/kg of distillate)', fontsize=12)
    fig.set_size_inches(3.25, 3.25)
    fig.savefig(map_dir + '/evaporator q vs P_b.png', bbox_inches='tight', dpi=300)
    fig.savefig(map_dir + '/evaporator q vs P_b.svg', bbox_inches='tight', dpi=300)

def make_maps_min_max_scale(map_dir, save_dir):
    var = 'LCOW'
    label = r'LCOW (\$/$m^3$ of product)'
    vmin = 3  # minimum cost on bar, $/m3
    vmax = 7  # maximum cost on bar, $/m3
    ticks = [3, 4, 5, 6, 7]  # tick marks on bar
    fmt = '.1f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'SEC'
    label = r'SEC (kWh/$m^3$ of product)'
    vmin = 15  # minimum cost on bar, $/m3
    vmax = 35  # maximum cost on bar, $/m3
    ticks = [15,20, 25,30,35]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'capex opex ratio'
    label = 'CAPEX/OPEX (-)'
    vmin = 0.55
    vmax = 0.75
    ticks = [0.55, 0.6, 0.65, 0.7,0.75]
    fmt = '.2f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Second law efficiency'
    var = 'Second law efficiency direct nayar'
    label = r'$\eta_{II}$ (%)'
    vmin = 4
    vmax = 20
    ticks = [4, 8, 12, 16, 20]
    fmt = '.1f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Brine temperature Celsius'
    label = 'Evaporator temperature (C)'
    vmin = 50  # minimum cost on bar, $/m3
    vmax = 100  # maximum cost on bar, $/m3
    ticks = [50, 75, 100]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Brine pressure kPa'
    label = 'Evaporator pressure (kPa)'
    vmin = 30  # minimum cost on bar, $/m3
    vmax = 38  # maximum cost on bar, $/m3
    ticks = [30, 32, 34, 36, 38]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'Compressor pressure ratio'
    label = r'Compressor pressure ratio (-)'
    vmin = 1.4  # minimum cost on bar, $/m3
    vmax = 2  # maximum cost on bar, $/m3
    ticks = [1.4,1.6,1.8,2]  # tick marks on bar
    fmt = '.1f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'Evaporator area'
    label = r'Evaporator area ($m^2$)'
    vmin = 400  # minimum cost on bar, $/m3
    vmax = 1200  # maximum cost on bar, $/m3
    ticks = [400,600,800,1000, 1200]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'Mass flux LMH'
    label = r'Product flux over evaporator (LMH)'
    vmin = 80  # minimum cost on bar, $/m3
    vmax = 160  # maximum cost on bar, $/m3
    ticks = [80, 100, 120, 140, 160]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    # Temperatures
    var = 'Compressed vapor temperature Celsius'
    label = 'Compressed vapor temperature (C)'
    vmin = 100
    vmax = 175
    ticks = [100, 125, 150, 175]
    fmt = '.0f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Distillate temperature Celsius'
    label = 'Distillate temperature (C)'
    vmin = 83  # minimum cost on bar, $/m3
    vmax = 87  # maximum cost on bar, $/m3
    ticks = [83, 84, 85, 86, 87]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Exiting distillate temperature Celsius'
    label = 'Exiting distillate temperature (C)'
    vmin = 32  # minimum cost on bar, $/m3
    vmax = 48  # maximum cost on bar, $/m3
    ticks = [32, 36, 40, 44, 48]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Exiting brine temperature Celsius'
    label = 'Exiting brine temperature (C)'
    vmin = 25  # minimum cost on bar, $/m3
    vmax = 40  # maximum cost on bar, $/m3
    ticks = [25, 30, 35, 40]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Preheated feed temperature Celsius'
    label = 'Preheated feed temperature (C)'
    vmin = 63  # minimum cost on bar, $/m3
    vmax = 73  # maximum cost on bar, $/m3
    ticks = [63, 65, 67, 69, 71, 73]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    # Distillate preheater
    var = 'Distillate hx area'
    label = r'Distillate HX area ($m^2$)'
    vmin = 100  # minimum cost on bar, $/m3
    vmax = 300  # maximum cost on bar, $/m3
    ticks = [100,150, 200, 250, 300]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Distillate hx LMTD'
    label = r'Distillate HX LMTD (K)'
    vmin = 5  # minimum cost on bar, $/m3
    vmax = 20  # maximum cost on bar, $/m3
    ticks = [5, 10, 15,20]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Distillate hx heat transfer kW'
    label = r'Distillate HX Q (kW)'
    vmin = 3000  # minimum cost on bar, $/m3
    vmax = 6000  # maximum cost on bar, $/m3
    ticks = [3000, 3750, 4500,5250, 6000]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Normalized distillate hx heat transfer kJ per kg'
    label = r'Distillate HX $\dot{Q}/\dot{m}_{feed}$ (kJ/kg-feed)'
    vmin = 150  # minimum cost on bar, $/m3
    vmax = 225  # maximum cost on bar, $/m3
    ticks = [150, 175, 200, 225]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    # Brine preheater
    var = 'Brine hx area'
    label = r'Brine HX area ($m^2$)'
    vmin = 25  # minimum cost on bar, $/m3
    vmax = 400  # maximum cost on bar, $/m3
    ticks = [25, 100, 175, 250, 325, 400]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'Brine hx LMTD'
    label = r'Brine HX LMTD (K)'
    vmin = 4  # minimum cost on bar, $/m3
    vmax = 16  # maximum cost on bar, $/m3
    ticks = [4, 8, 12, 16]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'Brine hx heat transfer kW'
    label = r'Brine HX Q (kW)'
    vmin = 800  # minimum cost on bar, $/m3
    vmax = 4300  # maximum cost on bar, $/m3
    ticks = [800, 1500, 2200, 2900, 3600,4300]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Normalized brine hx heat transfer kJ per kg'
    label = r'Brine HX $\dot{Q}/\dot{m}_{feed}$ (kJ/kg-feed)'
    vmin = 100  # minimum cost on bar, $/m3
    vmax = 175  # maximum cost on bar, $/m3
    ticks = [100, 125, 150, 175]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    # Compressor
    var = 'Normalized work kJ kg'
    label = 'Normalized compressor work\n(kJ/kg-product)'
    vmin = 45  # minimum cost on bar, $/m3
    vmax = 125  # maximum cost on bar, $/m3
    ticks = [45, 65, 85, 105, 125]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Compressor work kW'
    label = r'Compressor work (kW)'
    vmin = 900  # minimum cost on bar, $/m3
    vmax = 3600  # maximum cost on bar, $/m3
    ticks = [900, 1800, 2700, 3600]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Compressed vapor pressure kPa'
    label = r'Compressed vapor pressure (kPa)'
    vmin = 53  # minimum cost on bar, $/m3
    vmax = 61  # maximum cost on bar, $/m3
    ticks = [53, 55, 57, 59, 61]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    # Evaporator
    var = 'Evaporator LMTD'
    label = r'Evaporator LMTD (K)'
    vmin = 15  # minimum cost on bar, $/m3
    vmax = 40  # maximum cost on bar, $/m3
    ticks = [15, 20, 25, 30, 35, 40]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)

    var = 'Evaporator heat transfer MW'
    label = r'Evaporator Q (MW)'
    vmin = 35  # minimum cost on bar, $/m3
    vmax = 80  # maximum cost on bar, $/m3
    ticks = [35, 50, 65, 80]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)


def make_maps_final(map_dir, save_dir):
    var = 'capex opex ratio'
    label = 'CAPEX/OPEX (-)'
    vmin = 0
    vmax = 1
    ticks = [0,0.25, .5,0.75, 1]
    fmt = '.2f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)
    #
    var = 'Second law efficiency'
    var = 'Second law efficiency direct nayar'
    label = r'$\eta_{II}$ (%)'
    vmin = 0
    vmax = 20
    ticks = [0, 5,10,15,20]
    # ticks = [0, 20,40,60,80,100]
    fmt = '.1f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=False)
    # assert False
    var = 'LCOW'
    label = r'LCOW (\$/$m^3$ of product)'
    vmin = 0  # minimum cost on bar, $/m3
    vmax = 7  # maximum cost on bar, $/m3
    ticks = [0,1,2,3, 4, 5, 6,7]  # tick marks on bar
    fmt = '.1f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'SEC'
    label = r'SEC (kWh/$m^3$ of product)'
    vmin = 15  # minimum cost on bar, $/m3
    vmax = 35  # maximum cost on bar, $/m3
    ticks = [15,20, 25,30,35]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'Brine temperature Celsius'
    label = 'Evaporator temperature (C)'
    vmin = 50  # minimum cost on bar, $/m3
    vmax = 100  # maximum cost on bar, $/m3
    ticks = [50, 75, 100]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Brine pressure kPa'
    label = 'Evaporator pressure (kPa)'
    vmin = 0  # minimum cost on bar, $/m3
    vmax = 40  # maximum cost on bar, $/m3
    ticks = [0, 10,20,30,40]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'Evaporator area'
    label = r'Evaporator area ($m^2$)'
    vmin = 400  # minimum cost on bar, $/m3
    vmax = 1200  # maximum cost on bar, $/m3
    ticks = [400,600,800,1000, 1200]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'Compressed vapor temperature Celsius'
    label = 'Compressed vapor temperature (C)'
    vmin = 100
    vmax = 175
    fmt = '.0f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)

    var = 'Compressor pressure ratio'
    label = r'Compressor pressure ratio (-)'
    vmin = 1  # minimum cost on bar, $/m3
    vmax = 2  # maximum cost on bar, $/m3
    ticks = [1,1.25,1.50, 1.75,2]  # tick marks on bar
    fmt = '.2f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,make_ticks=False)


    var = 'Mass flux LMH'
    label = r'Product flux over evaporator (LMH)'
    vmin = 70  # minimum cost on bar, $/m3
    vmax = 150  # maximum cost on bar, $/m3
    ticks = [70, 90, 110, 130, 150]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

def make_maps(map_dir, save_dir):
    var = 'capex opex ratio'
    label = 'CAPEX/OPEX (-)'
    vmin = 0
    vmax = 1
    ticks = [0,0.25, .5, 0.75,1]
    fmt = '.2f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=True)

    var = 'LCOW normalized electricity'
    label = 'Electricty cost fraction (-)'
    vmin = 0
    vmax = 1
    ticks = [0,.5,1]
    fmt = '.2f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=True)

    var = 'LCOW normalized opex'
    label = 'Operating cost fraction (-)'
    vmin = 0
    vmax = 1
    ticks = [0, .5, 1]
    fmt = '.2f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=True)

    var = 'Distillate hx LMTD'
    label = r'Distillate HX LMTD (K)'
    vmin = 0
    vmax = 10
    ticks = [0, 10]
    fmt = '.0f'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=True)

    var = 'Brine hx LMTD'
    label = r'Brine HX LMTD (K)'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=True)

    var = 'Distillate hx heat transfer kW'
    label = r'Distillate HX Q (kW)'
    vmin = 0
    vmax = 10
    ticks = [0,10]
    fmt = '.0f'
    plot_2D_heat_map(map_dir, save_dir, var,label, vmin, vmax, ticks,fmt, make_ticks=True)

    var = 'Brine hx heat transfer kW'
    label = r'Brine HX Q (kW)'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=True)

    var = 'Normalized distillate hx heat transfer kJ per kg'
    label = r'Distillate HX normalized q [kJ/kg-feed]'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=True)

    var = 'Normalized brine hx heat transfer kJ per kg'
    label = r'Brine HX normalized q [kJ/kg-feed]'
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, make_ticks=True)

    var = 'LCOW'
    label = r'LCOW (\$/$m^3$ of product)'
    vmin = 2.8  # minimum cost on bar, $/m3
    vmax = 6.1  # maximum cost on bar, $/m3
    ticks = [3, 4, 5, 6]  # tick marks on bar
    fmt = '.1f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'SEC'
    label = r'SEC (kWh/$m^3$ of product)'
    vmin = 15  # minimum cost on bar, $/m3
    vmax = 63  # maximum cost on bar, $/m3
    ticks = [20, 30, 40, 50, 63]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Brine temperature Celsius'
    label = 'Evaporator temperature (C)'
    vmin = 25  # minimum cost on bar, $/m3
    vmax = 150  # maximum cost on bar, $/m3
    ticks = [25, 50, 100, 150]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Brine pressure kPa'
    label = 'Evaporator pressure (kPa)'
    vmin = 0  # minimum cost on bar, $/m3
    vmax = 403  # maximum cost on bar, $/m3
    ticks = [0, 100, 200, 300, 400]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Preheated feed temperature Celsius'
    label = 'Preheated feed temperature (C)'
    vmin = 25  # minimum cost on bar, $/m3
    vmax = 150  # maximum cost on bar, $/m3
    ticks = [50, 100, 150]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Compressed vapor temperature Celsius'
    label = 'Compressed vapor temperature (C)'
    vmin = 87  # minimum cost on bar, $/m3
    vmax = 227  # maximum cost on bar, $/m3
    ticks = [100, 150, 200]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Compressed vapor pressure kPa'
    label = 'Compressed vapor pressure (kPa)'
    vmin = 3  # minimum cost on bar, $/m3
    vmax = 629  # maximum cost on bar, $/m3
    ticks = [200, 400, 600]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Distillate temperature Celsius'
    label = 'Distillate temperature (C)'
    vmin = 34  # minimum cost on bar, $/m3
    vmax = 161  # maximum cost on bar, $/m3
    ticks = [50, 100, 150]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Preheater split ratio'
    label = 'Preheater split ratio (-)'
    vmin = 0.4  # minimum cost on bar, $/m3
    vmax = 1  # maximum cost on bar, $/m3
    ticks = [0.4, 0.6, 0.8]  # tick marks on bar
    fmt = '.2f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Distillate hx area'
    label = r'Distillate preheater area ($\rmm^2$)'
    vmin = 0  # minimum cost on bar, $/m3
    vmax = 1630  # maximum cost on bar, $/m3
    ticks = [500, 1000, 2000, 3000, 4000]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Brine hx area'
    label = r'Brine preheater area ($\rmm^2$)'
    vmin = 0  # minimum cost on bar, $/m3
    vmax = 1001  # maximum cost on bar, $/m3
    ticks = [250, 500, 750]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Evaporator area'
    label = r'Evaporator area ($\rmm^2$)'
    vmin = 742  # minimum cost on bar, $/m3
    vmax = 3740  # maximum cost on bar, $/m3
    ticks = [1000, 2000, 3000]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Evaporator LMTD'
    label = r'Evaporator LMTD (K)'
    vmin = 19  # minimum cost on bar, $/m3
    vmax = 61  # maximum cost on bar, $/m3
    ticks = [20, 40, 60]  # tick marks on bar
    fmt = '.1f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Compressor pressure ratio'
    label = r'Compressor pressure ratio (-)'
    vmin = 1.5  # minimum cost on bar, $/m3
    vmax = 3.6  # maximum cost on bar, $/m3
    ticks = [2, 2.5, 3]  # tick marks on bar
    fmt = '.2f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Evaporator-feed temperature difference'
    label = r'$T_{brine}-T_{feed}$ (C)'
    vmin = -7  # minimum cost on bar, $/m3
    vmax = 15  # maximum cost on bar, $/m3
    ticks = [-5, 0, 5, 10, 15]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Mass flux LMH'
    label = r'Product flux over evaporator (LMH)'
    vmin = 28  # minimum cost on bar, $/m3
    vmax = 82  # maximum cost on bar, $/m3
    ticks = [30, 40, 50, 60, 70, 80]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

def make_maps_dual_param(map_dir, save_dir,xticklabels,yticklabels,xlabel,ylabel):
    var = 'LCOW'
    label = r'LCOW (\$/$\rmm^3$ of product)'
    vmin = 2.8  # minimum cost on bar, $/m3
    vmax = 6.1  # maximum cost on bar, $/m3
    ticks = [3, 4, 5, 6]  # tick marks on bar
    fmt = '.1f'  # format of annotation
    # plot_2D_heat_map_dual_param(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt,xticklabels,yticklabels,xlabel,ylabel)
    vmin = -25
    vmax = 25
    ticks = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
    fmt = '.0f'
    plot_2D_heat_map_dual_param(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, xticklabels, yticklabels,xlabel,ylabel, make_ticks=False, dif=True)

    var = 'SEC'
    label = r'SEC (kWh/$\rmm^3$ of product)'
    vmin = -25  # minimum cost on bar, $/m3
    vmax = 25  # maximum cost on bar, $/m3
    ticks = [-25,-20,-15,-10,-5,0,5,10,15,20,25]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    # plot_2D_heat_map_dual_param(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, xticklabels, yticklabels, xlabel,
    #                             ylabel)
    # vmin = -10.8
    # vmax = 10.8
    # ticks = [-10,-5,0,5,10]
    plot_2D_heat_map_dual_param(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, xticklabels, yticklabels, xlabel,
                                ylabel, make_ticks=False,dif=True)
    # plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'capex opex ratio'
    label = 'CAPEX/OPEX (-)'
    vmin = 0
    vmax = 1
    ticks = [0, .5, 1]
    fmt = '.2f'
    # plot_2D_heat_map_dual_param(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, xticklabels, yticklabels, xlabel,
    #                             ylabel)
    vmin = -2
    vmax = 2
    ticks = [-2,-1,0,1,2]
    fmt = '.0f'
    plot_2D_heat_map_dual_param(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt, xticklabels, yticklabels, xlabel,
                                ylabel, make_ticks=False, dif=True)
    assert False
    var = 'Brine temperature Celsius'
    label = 'Evaporator temperature [C]'
    vmin = 25  # minimum cost on bar, $/m3
    vmax = 150  # maximum cost on bar, $/m3
    ticks = [25, 50, 100, 150]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Brine pressure kPa'
    label = 'Evaporator pressure [kPa]'
    vmin = 0  # minimum cost on bar, $/m3
    vmax = 403  # maximum cost on bar, $/m3
    ticks = [0, 100, 200, 300, 400]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Preheated feed temperature Celsius'
    label = 'Preheated feed temperature [C]'
    vmin = 25  # minimum cost on bar, $/m3
    vmax = 150  # maximum cost on bar, $/m3
    ticks = [50, 100, 150]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Compressed vapor temperature Celsius'
    label = 'Compressed vapor temperature [C]'
    vmin = 87  # minimum cost on bar, $/m3
    vmax = 227  # maximum cost on bar, $/m3
    ticks = [100, 150, 200]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Compressed vapor pressure kPa'
    label = 'Compressed vapor pressure [kPa]'
    vmin = 3  # minimum cost on bar, $/m3
    vmax = 629  # maximum cost on bar, $/m3
    ticks = [200, 400, 600]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Distillate temperature Celsius'
    label = 'Distillate temperature [C]'
    vmin = 34  # minimum cost on bar, $/m3
    vmax = 161  # maximum cost on bar, $/m3
    ticks = [50, 100, 150]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Preheater split ratio'
    label = 'Preheater split ratio [C]'
    vmin = 0.4  # minimum cost on bar, $/m3
    vmax = 1  # maximum cost on bar, $/m3
    ticks = [0.4, 0.6, 0.8]  # tick marks on bar
    fmt = '.2f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Distillate hx area'
    label = r'Distillate preheater area [$\rmm^2$]'
    vmin = 0  # minimum cost on bar, $/m3
    vmax = 1630  # maximum cost on bar, $/m3
    ticks = [500, 1000, 2000, 3000, 4000]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Brine hx area'
    label = r'Brine preheater area [$\rmm^2$]'
    vmin = 0  # minimum cost on bar, $/m3
    vmax = 1001  # maximum cost on bar, $/m3
    ticks = [250, 500, 750]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Evaporator area'
    label = r'Evaporator area [$\rmm^2$]'
    vmin = 742  # minimum cost on bar, $/m3
    vmax = 3740  # maximum cost on bar, $/m3
    ticks = [1000, 2000, 3000]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Evaporator LMTD'
    label = r'Evaporator LMTD [K]'
    vmin = 19  # minimum cost on bar, $/m3
    vmax = 61  # maximum cost on bar, $/m3
    ticks = [20, 40, 60]  # tick marks on bar
    fmt = '.1f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Compressor pressure ratio'
    label = r'Compressor pressure ratio [-]'
    vmin = 1.5  # minimum cost on bar, $/m3
    vmax = 3.6  # maximum cost on bar, $/m3
    ticks = [2, 2.5, 3]  # tick marks on bar
    fmt = '.2f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Evaporator-feed temperature difference'
    label = r'$T_{brine}-T_{feed}$ [C]'
    vmin = -7  # minimum cost on bar, $/m3
    vmax = 15  # maximum cost on bar, $/m3
    ticks = [-5, 0, 5, 10, 15]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

    var = 'Mass flux LMH'
    label = r'Product flux over evaporator [LMH]'
    vmin = 28  # minimum cost on bar, $/m3
    vmax = 82  # maximum cost on bar, $/m3
    ticks = [30, 40, 50, 60, 70, 80]  # tick marks on bar
    fmt = '.0f'  # format of annotation
    plot_2D_heat_map(map_dir, save_dir, var, label, vmin, vmax, ticks, fmt)

def tornado_plot(map_dir, save_dir):
    pos = '..'
    neg = 'xx'
    df = pd.read_csv(map_dir+'tornado_results.csv')
    labels_dict = {}
    labels_dict['compressor_cost'] = 'Compressor Cost'
    labels_dict['compressor_efficiency'] = 'Compressor Efficiency'
    labels_dict['electricity_cost'] = 'Electricity Cost'
    labels_dict['U_evap'] = 'Evaporator U'
    labels_dict['U_hx'] = 'Preheater U'
    labels_dict['material_factor'] = 'Material Factor'
    labels_dict['T_b'] = 'Evaporator Temperature'

    n_param = len(labels_dict.keys())
    # n_param = 7
    widths = {}
    starts = {}
    colors = {}
    labels = []
    min_LCOW = min(min(df['LCOW_min_per']), min(df['LCOW_max_per']))
    max_LCOW = max(max(df['LCOW_min_per']), max(df['LCOW_max_per']))
    min_SEC = min(min(df['SEC_min_per']), min(df['SEC_max_per']))
    max_SEC = max(max(df['SEC_min_per']), max(df['SEC_max_per']))
    min_co_ratio = min(min(df['capex_opex_ratio_min_per']), min(df['capex_opex_ratio_max_per']))
    max_co_ratio = max(max(df['capex_opex_ratio_min_per']), max(df['capex_opex_ratio_max_per']))

    widths['LCOW'] = []
    widths['SEC'] = []
    widths['co_ratio'] = []
    starts['LCOW'] = []
    starts['SEC'] = []
    starts['co_ratio'] = []
    colors['LCOW'] = []
    colors['SEC'] = []
    colors['co_ratio'] = []
    for i in range(n_param):
        labels.append(labels_dict[df['parameter'][i]])
        if df['LCOW_min_per'][i] < df['LCOW_max_per'][i]:
            colors['LCOW'].append(pos)
        else:
            colors['LCOW'].append(neg)
        min_lcow = min(df['LCOW_min_per'][i],df['LCOW_max_per'][i])
        max_lcow = max(df['LCOW_min_per'][i],df['LCOW_max_per'][i])
        widths['LCOW'].append(max_lcow-min_lcow)
        starts['LCOW'].append(min_lcow)

        if df['SEC_min_per'][i] < df['SEC_max_per'][i]:
            colors['SEC'].append(pos)
        else:
            colors['SEC'].append(neg)
        min_sec = min(df['SEC_min_per'][i], df['SEC_max_per'][i])
        max_sec = max(df['SEC_min_per'][i], df['SEC_max_per'][i])
        widths['SEC'].append(max_sec - min_sec)
        starts['SEC'].append(min_sec)

        if df['capex_opex_ratio_min_per'][i] < df['capex_opex_ratio_max_per'][i]:
            colors['co_ratio'].append(pos)
        else:
            colors['co_ratio'].append(neg)
        min_cor = min(df['capex_opex_ratio_min_per'][i], df['capex_opex_ratio_max_per'][i])
        max_cor = max(df['capex_opex_ratio_min_per'][i], df['capex_opex_ratio_max_per'][i])
        widths['co_ratio'].append(max_cor - min_cor)
        starts['co_ratio'].append(min_cor)

    # Reorder to plot in descending order by LCOW
    idx = sorted(range(len(widths['LCOW'])), key=lambda index: widths['LCOW'][index])
    widths['LCOW_descending'] = sorted(widths['LCOW'])
    starts['LCOW_descending'] = []
    widths['SEC_reorder'] = []
    widths['co_ratio_reorder'] = []
    starts['SEC_reorder'] = []
    starts['co_ratio_reorder'] = []
    labels_reorder =[]
    hatch = {}
    hatch['LCOW'] = []
    hatch['SEC'] = []
    hatch['co_ratio']=[]
    for i in idx:
        starts['LCOW_descending'].append(starts['LCOW'][i])
        labels_reorder.append(labels[i])
        widths['SEC_reorder'].append(widths['SEC'][i])
        widths['co_ratio_reorder'].append(widths['co_ratio'][i])
        starts['SEC_reorder'].append(starts['SEC'][i])
        starts['co_ratio_reorder'].append(starts['co_ratio'][i])
        hatch['LCOW'].append(colors['LCOW'][i])
        hatch['SEC'].append(colors['SEC'][i])
        hatch['co_ratio'].append(colors['co_ratio'][i])


    fig = plt.figure()
    ax = fig.axes
    plt.barh(labels_reorder, widths['LCOW_descending'], left=starts['LCOW_descending'],hatch=hatch['LCOW'])
    plt.axvline(0,linestyle='--', color='black')
    plt.xlabel('Percentage Change in LCOW (%)',fontsize=10)
    plt.xlim(-50,50)
    plt.xticks([-50, -40,-30,-20,-10,0,10,20,30,40,50])
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.axes
    plt.barh(labels_reorder, widths['SEC_reorder'], left=starts['SEC_reorder'],hatch=hatch['SEC'])
    plt.axvline(0, linestyle='--', color='black')
    plt.xlabel('Percentage Change in SEC (%)',fontsize=10)
    plt.xlim(-50,50)
    plt.xticks([-50, -40,-30,-20,-10,0,10,20,30,40,50])
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.axes
    plt.barh(labels_reorder, widths['co_ratio_reorder'], left=starts['co_ratio_reorder'],hatch=hatch['co_ratio'])
    plt.axvline(0,linestyle='--', color='black')
    plt.xlabel('Percentage Change in CAPEX/OPEX (%)',fontsize=10)
    plt.xlim(-50,50)
    plt.xticks([-50, -40,-30,-20,-10,0,10,20,30,40,50])
    plt.tight_layout()
    plt.show()

def tornado_plot_multiple(map_dir,cases):
    n_cases = len(cases)
    pos = ''
    neg = '//'
    # base ordering on case 1
    labels_dict = {}
    labels_dict['compressor_cost'] = 'Compressor Cost'
    labels_dict['compressor_efficiency'] = 'Compressor Efficiency'
    labels_dict['electricity_cost'] = 'Electricity Cost'
    labels_dict['U_evap'] = 'Evaporator Overall\nHeat Transfer Coefficient'
    labels_dict['U_hx'] = 'Preheater Overall\nHeat Transfer Coefficient'
    labels_dict['material_factor'] = 'Material Factor'
    labels_dict['T_b'] = 'Evaporator Temperature'
    labels_dict['material_factor_50per'] = 'Material Factor'

    n_param = len(labels_dict.keys())-1

    for name,case in cases.items():
        df = pd.read_csv(map_dir + name +'/tornado_results.csv') #_fm50

        widths = {}
        starts = {}
        colors = {}
        labels = []
        min_LCOW = min(min(df['LCOW_min_per']), min(df['LCOW_max_per']))
        max_LCOW = max(max(df['LCOW_min_per']), max(df['LCOW_max_per']))
        min_SEC = min(min(df['SEC_min_per']), min(df['SEC_max_per']))
        max_SEC = max(max(df['SEC_min_per']), max(df['SEC_max_per']))
        min_co_ratio = min(min(df['capex_opex_ratio_min_per']), min(df['capex_opex_ratio_max_per']))
        max_co_ratio = max(max(df['capex_opex_ratio_min_per']), max(df['capex_opex_ratio_max_per']))

        widths['LCOW'] = []
        widths['SEC'] = []
        widths['co_ratio'] = []
        starts['LCOW'] = []
        starts['SEC'] = []
        starts['co_ratio'] = []
        colors['LCOW'] = []
        colors['SEC'] = []
        colors['co_ratio'] = []
        for i in range(n_param):
            labels.append(labels_dict[df['parameter'][i]])
            if df['LCOW_min_per'][i] < df['LCOW_max_per'][i]:
                colors['LCOW'].append(pos)
            else:
                colors['LCOW'].append(neg)
            min_lcow = min(df['LCOW_min_per'][i], df['LCOW_max_per'][i])
            max_lcow = max(df['LCOW_min_per'][i], df['LCOW_max_per'][i])
            widths['LCOW'].append(max_lcow - min_lcow)
            starts['LCOW'].append(min_lcow)

            if df['SEC_min_per'][i] < df['SEC_max_per'][i]:
                colors['SEC'].append(pos)
            else:
                colors['SEC'].append(neg)
            min_sec = min(df['SEC_min_per'][i], df['SEC_max_per'][i])
            max_sec = max(df['SEC_min_per'][i], df['SEC_max_per'][i])
            widths['SEC'].append(max_sec - min_sec)
            starts['SEC'].append(min_sec)

            if df['capex_opex_ratio_min_per'][i] < df['capex_opex_ratio_max_per'][i]:
                colors['co_ratio'].append(pos)
            else:
                colors['co_ratio'].append(neg)
            min_cor = min(df['capex_opex_ratio_min_per'][i], df['capex_opex_ratio_max_per'][i])
            max_cor = max(df['capex_opex_ratio_min_per'][i], df['capex_opex_ratio_max_per'][i])
            widths['co_ratio'].append(max_cor - min_cor)
            starts['co_ratio'].append(min_cor)

        case['widths'] = widths
        case['starts'] = starts
        case['colors'] = colors


    # Reorder to plot in descending order by LCOW based on case 1
    idx = sorted(range(len(cases['case_1']['widths']['LCOW'])), key=lambda index: cases['case_1']['widths']['LCOW'][index])
    cases['case_1']['widths']['LCOW_descending'] = sorted(cases['case_1']['widths']['LCOW'])

    for name,case in cases.items():
        case['starts']['LCOW_descending'] = []
        case['widths']['SEC_reorder'] = []
        case['widths']['co_ratio_reorder'] = []
        case['starts']['SEC_reorder'] = []
        case['starts']['co_ratio_reorder'] = []
        labels_reorder =[]
        case['hatch'] = {}
        case['hatch']['LCOW'] = []
        case['hatch']['SEC'] = []
        case['hatch']['co_ratio']=[]
        for i in idx:
            case['starts']['LCOW_descending'].append(case['starts']['LCOW'][i])
            labels_reorder.append(labels[i])
            case['widths']['SEC_reorder'].append(case['widths']['SEC'][i])
            case['widths']['co_ratio_reorder'].append(case['widths']['co_ratio'][i])
            case['starts']['SEC_reorder'].append(case['starts']['SEC'][i])
            case['starts']['co_ratio_reorder'].append(case['starts']['co_ratio'][i])
            case['hatch']['LCOW'].append(case['colors']['LCOW'][i])
            case['hatch']['SEC'].append(case['colors']['SEC'][i])
            case['hatch']['co_ratio'].append(case['colors']['co_ratio'][i])
        if name != 'case_1':
            case['widths']['LCOW_descending'] = []
            for i in idx:
                case['widths']['LCOW_descending'].append(case['widths']['LCOW'][i])

    pos_patch = mpatches.Patch(facecolor='none',edgecolor='k', label='Positive')
    neg_patch = mpatches.Patch(facecolor='none', edgecolor='k',hatch=neg,label='Negative')
    case1_patch = mpatches.Patch(color=cases['case_1']['color'], edgecolor='k',label='Case 1')
    case2_patch = mpatches.Patch(color=cases['case_2']['color'], edgecolor='k',label='Case 2')
    case3_patch = mpatches.Patch(color=cases['case_3']['color'], edgecolor='k',label='Case 3')


    fig,ax = plt.subplots()
    width = 0.3
    for i in range(n_param):
        ax.barh(i+width, cases['case_1']['widths']['LCOW_descending'][i], width,left=cases['case_1']['starts']['LCOW_descending'][i],color=cases['case_1']['color'],hatch=cases['case_1']['hatch']['LCOW'][i],edgecolor = cases['case_1']['edgecolor'],label='Case 1', linewidth=1.5)
        ax.barh(i, cases['case_2']['widths']['LCOW_descending'][i], width,left=cases['case_2']['starts']['LCOW_descending'][i],color=cases['case_2']['color'],hatch=cases['case_2']['hatch']['LCOW'][i],edgecolor = cases['case_2']['edgecolor'],label='Case 2', linewidth=1.5)
        ax.barh(i-width, cases['case_3']['widths']['LCOW_descending'][i], width,left=cases['case_3']['starts']['LCOW_descending'][i],color=cases['case_3']['color'],hatch=cases['case_3']['hatch']['LCOW'][i],edgecolor = cases['case_3']['edgecolor'],label='Case 3', linewidth=1.5)
    # ax.set(yticks=np.arange(n_param)+width,yticklabels=labels_reorder)
    ax.set_yticks(np.arange(n_param))
    ax.set_yticklabels(labels_reorder, fontsize=14)
    # ax.set_xticks(fontsize=10)
    ax.legend(handles=[pos_patch,neg_patch, case1_patch, case2_patch, case3_patch], frameon=False, loc='lower right', fontsize=12)
    plt.axvline(0,linestyle='--', color='black')

    plt.xlabel('Percentage Change in LCOW (%)',fontsize=14)
    plt.xlim(-25,25)
    plt.xticks([-25, -20,-15,-10,-5,0,5,10,15,20,25],fontsize=10)
    plt.tight_layout()
    plt.show()
    # assert False

    fig,ax = plt.subplots()
    width = 0.3
    for i in range(n_param):
        ax.barh(i+width, cases['case_1']['widths']['SEC_reorder'][i], width,left=cases['case_1']['starts']['SEC_reorder'][i],color=cases['case_1']['color'],hatch=cases['case_1']['hatch']['SEC'][i],edgecolor = cases['case_1']['edgecolor'],label='Case 1', linewidth=1.5)
        ax.barh(i, cases['case_2']['widths']['SEC_reorder'][i], width,left=cases['case_2']['starts']['SEC_reorder'][i],color=cases['case_2']['color'],hatch=cases['case_2']['hatch']['SEC'][i],edgecolor = cases['case_2']['edgecolor'],label='Case 2', linewidth=1.5)
        ax.barh(i-width, cases['case_3']['widths']['SEC_reorder'][i], width,left=cases['case_3']['starts']['SEC_reorder'][i],color=cases['case_3']['color'],hatch=cases['case_3']['hatch']['SEC'][i],edgecolor = cases['case_3']['edgecolor'],label='Case 3', linewidth=1.5)
    # ax.set(yticks=np.arange(n_param)+width,yticklabels=labels_reorder)
    ax.set_yticks(np.arange(n_param))
    ax.set_yticklabels(labels_reorder, fontsize=14)
    # ax.legend(['Case 1','Case 2', 'Case 3'],frameon=False,loc='lower right')
    ax.legend(handles=[pos_patch,neg_patch, case1_patch, case2_patch, case3_patch], frameon=False, loc='lower right', fontsize=12)
    plt.axvline(0,linestyle='--', color='black')
    plt.xlabel('Percentage Change in SEC (%)',fontsize=14)
    plt.xlim(-25,25)
    plt.xticks([-25, -20,-15,-10,-5,0,5,10,15,20,25], fontsize=10)
    plt.tight_layout()
    plt.show()
    # assert False


    fig,ax = plt.subplots()
    # ax = fig.axes
    width = 0.3
    for i in range(n_param):
        ax.barh(i+width, cases['case_1']['widths']['co_ratio_reorder'][i], width,left=cases['case_1']['starts']['co_ratio_reorder'][i],color=cases['case_1']['color'],hatch=cases['case_1']['hatch']['co_ratio'][i],edgecolor = cases['case_1']['edgecolor'],label='Case 1', linewidth=1.5)
        ax.barh(i, cases['case_2']['widths']['co_ratio_reorder'][i], width,left=cases['case_2']['starts']['co_ratio_reorder'][i],color=cases['case_2']['color'],hatch=cases['case_2']['hatch']['co_ratio'][i],edgecolor = cases['case_2']['edgecolor'],label='Case 2', linewidth=1.5)
        ax.barh(i-width, cases['case_3']['widths']['co_ratio_reorder'][i], width,left=cases['case_3']['starts']['co_ratio_reorder'][i],color=cases['case_3']['color'],hatch=cases['case_3']['hatch']['co_ratio'][i],edgecolor = cases['case_3']['edgecolor'],label='Case 3', linewidth=1.5)
    # ax.set(yticks=np.arange(n_param)+width,yticklabels=labels_reorder)
    ax.set_yticks(np.arange(n_param))
    ax.set_yticklabels(labels_reorder, fontsize=14)
    ax.legend(handles=[pos_patch,neg_patch, case1_patch, case2_patch, case3_patch], frameon=False, loc='lower right', fontsize=12)
    # ax.legend(['Case 1','Case 2', 'Case 3'],frameon=False,loc='lower right')
    plt.axvline(0,linestyle='--', color='black')
    plt.xlabel('Percentage Change in CAPEX/OPEX (%)',fontsize=14)
    plt.xlim(-25,25)
    plt.xticks([-25, -20,-15,-10,-5,0,5,10,15,20,25], fontsize=10)
    plt.tight_layout()
    plt.show()

def save_tornado_plot_data(map_dir,LCOW_base=None,SEC_base=None,co_ratio_base=None):
    outputs = {}
    outputs['parameter'] = []
    outputs['param_min'] = []
    outputs['param_max'] = []
    outputs['LCOW_min'] = []
    outputs['LCOW_max'] = []
    outputs['SEC_min'] = []
    outputs['SEC_max'] = []
    outputs['capex_opex_ratio_min'] = []
    outputs['capex_opex_ratio_max'] = []

    # Baseline case for 100 g/kg, 50% recovery
    if LCOW_base is None:
        print('No LCOW base value given - using 100 g/kg, 50% recovery values')
        LCOW_base = 6.329 #5.39
        SEC_base = 29.228#33.5
        co_ratio_base = 0.7065 #0.435/0.565

    parameters = ['electricity_cost',
                  'material_factor',
                  "U_evap",
                  "U_hx",
                  'T_b',
                  'compressor_efficiency',
                  'compressor_cost']

    for filename in os.listdir(map_dir):
        # print(filename)
        param_name = filename.split('.')[0]
        if param_name not in parameters:
            continue
        # if param_name == 'figures':
        #     continue
        # if param_name == 'tornado_results':
        #     continue
        # if param_name == 'tornado_results_with_elec':
        #     continue
        print(param_name)
        outputs['parameter'].append(param_name)
        df = pd.read_csv(map_dir+filename)
        if param_name == 'material_factor_50per':
            param_name = 'material_factor'
        outputs['param_min'].append(df['# '+param_name][0])
        outputs['param_max'].append(df['# '+param_name][1])
        outputs['LCOW_min'].append(df['LCOW'][0])
        outputs['LCOW_max'].append(df['LCOW'][1])
        outputs['SEC_min'].append(df['SEC'][0])
        outputs['SEC_max'].append(df['SEC'][1])
        outputs['capex_opex_ratio_min'].append(df['capex opex ratio'][0])
        outputs['capex_opex_ratio_max'].append(df['capex opex ratio'][1])

    outputs['LCOW_min_per'] = (np.array(outputs['LCOW_min'])/LCOW_base-1)*100
    outputs['LCOW_max_per'] = (np.array(outputs['LCOW_max'])/LCOW_base-1)*100
    outputs['SEC_min_per'] = (np.array(outputs['SEC_min'])/SEC_base-1)*100
    outputs['SEC_max_per'] = (np.array(outputs['SEC_max'])/SEC_base-1)*100
    outputs['capex_opex_ratio_min_per'] = (np.array(outputs['capex_opex_ratio_min'])/co_ratio_base-1)*100
    outputs['capex_opex_ratio_max_per'] = (np.array(outputs['capex_opex_ratio_max'])/co_ratio_base-1)*100

    # save to csv
    df = pd.DataFrame(outputs)
    df.to_csv(map_dir+'tornado_results.csv', index=False)

if __name__ == "__main__":
    main()
