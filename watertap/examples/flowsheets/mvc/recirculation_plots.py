import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np


def main():
    # plot_rr_sys_rr_evap(wf=0.1,rr=np.arange(0.2,0.8,0.1), beta=[0,0.1,0.3,0.5,0.7])
    # plot_wf_sys_wf_evap(wf=np.arange(0.025,0.15,0.025), rr=0.5,beta=[0,0.1,0.3,0.5,0.7])
    # plot_W_least_vs_recovery(wf=0.05, rr=np.arange(0.2,0.6,0.1),beta=[0.1,0.3,0.5,0.7])
    plot_W_least_vs_wf(wf=np.arange(0.025,0.15,0.025), rr=0.4,beta=[0.1,0.3,0.5,0.7])
    # plot_W_least_vs_wf(wf=np.array([0.025,0.05]), rr=0.4,beta=[0.1,0.3])

    # plot_LCOW_SEC_vs_rr()

def get_W_min_vs_wf_rr():
    wf = [25,50,75,100,125,150]
    return

def plot_LCOW_SEC_vs_rr():
    rr = [30,35,40,45,50,55,60,65]
    beta = [10,30,50,70]
    fig = plt.figure()
    filename = "C:/Users/carso/Documents/MVC/watertap_results/mvc_recirculation/brine_recirc_"
    for i,b in enumerate(beta):
        results = pd.read_csv(filename+str(b)+'.csv')
        plt.plot(rr,results['LCOW'],label=str(b))

    plt.legend(frameon=False, title='Portion of brine\nrecirculated',fontsize=8)
    plt.xlim([min(rr), max(rr)])
    plt.xlabel('Recovery (-)')
    plt.ylabel('LCOW ($/m3)')
    fig.set_size_inches(3.25,3.25)
    plt.title(r'$w_f$ = 100 g/kg')
    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    for i,b in enumerate(beta):
        results = pd.read_csv(filename+str(b)+'.csv')
        plt.plot(rr,results['SEC'],label=str(b))

    plt.legend(frameon=False, title='Portion of brine\nrecirculated',fontsize=8)
    plt.xlim([min(rr), max(rr)])
    plt.xlabel('Recovery (-)')
    plt.ylabel('SEC (kWh/m3)')
    fig.set_size_inches(3.25,3.25)
    plt.title(r'$w_f$ = 100 g/kg')
    fig.tight_layout()
    plt.show()

def plot_rr_sys_rr_evap(wf=0.1, rr=[0.5],beta=[0.1]):
    mf=1
    n_rr = len(rr)
    n_beta = len(beta)
    rr_evap = np.zeros((n_rr, n_beta))
    wf_evap = np.zeros((n_rr, n_beta))

    for i,r in enumerate(rr):
        mv = r*mf
        mb = (1-r)*mf
        wb = wf/(1-r)
        for j,b in enumerate(beta):
            mb_evap = mb/(1-b)
            mb_rec = mb_evap-mb
            mf_evap = mb_rec + mf
            rr_evap[i,j] = mv/mf_evap
            wf_evap[i,j] = wb*mb_evap/mf_evap

    print(rr_evap)
    fig = plt.figure()
    for j,b in enumerate(beta):
        plt.plot(rr, rr_evap[:,j],label=str(b))
    plt.legend(frameon=False, title='Portion of brine\nrecirculated',fontsize=8)
    plt.xlim([min(rr), max(rr)])
    plt.xlabel('System recovery (-)')
    plt.ylabel('Evaporator recovery (-)')
    fig.set_size_inches(3.25,3.25)
    plt.title(r'$w_f$ = 100 g/kg')
    fig.tight_layout()
    plt.show()
    return

def plot_wf_sys_wf_evap(wf=0.1, rr=0.5,beta=[0.1]):
    mf=1
    n_wf = len(wf)
    n_beta = len(beta)
    rr_evap = np.zeros((n_wf, n_beta))
    wf_evap = np.zeros((n_wf, n_beta))

    for i,w in enumerate(wf):
        mv = rr*mf
        mb = (1-rr)*mf
        wb = w/(1-rr)
        for j,b in enumerate(beta):
            mb_evap = mb/(1-b)
            mb_rec = mb_evap-mb
            mf_evap = mb_rec + mf
            rr_evap[i,j] = mv/mf_evap
            wf_evap[i,j] = wb*mb_evap/mf_evap

    print(rr_evap)
    fig = plt.figure()
    for j,b in enumerate(beta):
        plt.plot(wf*1000, wf_evap[:,j]*1000,label=str(b))
    plt.legend(frameon=False, title='Portion of brine\nrecirculated',fontsize=8)
    plt.xlim([min(wf*1000), max(wf*1000)])
    plt.xlabel('Feed salinity (g/kg)')
    plt.ylabel('Salinity of feed entering evaporator (g/kg)')
    plt.title('Recovery = 50%')
    fig.set_size_inches(3.25,3.25)
    fig.tight_layout()
    plt.show()
    return

def plot_W_least_vs_recovery(wf=0.05, rr=[0.5], beta=[0.1]):
    T_f = 342.15
    T_f_recirc = 342.96
    T_b = 348.15
    T_b = T_b
    mf = 1
    n_rr = len(rr)
    n_beta = len(beta)
    rr_evap = np.zeros((n_rr, n_beta))
    wf_evap = np.zeros((n_rr, n_beta))
    W_least = np.zeros((n_rr, n_beta))
    W_least_with_recirc = np.zeros((n_rr, n_beta))

    for i,r in enumerate(rr):
        mv = r*mf
        mb = (1-r)*mf
        wb = wf/(1-r)
        for j,b in enumerate(beta):
            mb_evap = mb/(1-b)
            mb_rec = mb_evap-mb
            mf_evap = mb_rec + mf
            rr_evap[i,j] = mv/mf_evap
            wf_evap[i,j] = wb*mb_evap/mf_evap
            W_least[i,j] = get_W_least(r,mf,mb,mv,wf,wb,T_f,T_b)
            W_least_with_recirc[i,j] = get_W_least(rr_evap[i,j],mf_evap, mb_evap,mv,wf_evap[i,j],wb,T_f_recirc,T_b)

    print(W_least-W_least_with_recirc)
    fig = plt.figure()
    for j,b in enumerate(beta):
        plt.plot(rr, ((W_least_with_recirc[:,j]/W_least[:,j])-1)*100,label=str(b))
    plt.legend(frameon=False, title='Portion of brine\nrecirculated',fontsize=8)
    plt.xlim([min(rr), max(rr)])
    plt.xlabel('System recovery (-)')
    plt.ylabel(r'$(W_{least,recirc} - W_{least})/W_{least}\times 100$ (%)')
    fig.set_size_inches(3.25,3.25)
    plt.title(r'$w_f$ = '+ str(wf*1e3) +' g/kg')
    fig.tight_layout()
    plt.show()
    return


def plot_W_least_vs_wf(wf=[0.05], rr=0.5, beta=[0.1]):
    T_f = 342.15
    T_f_recirc = 342.15
    # T_f_recirc = 342.96
    T_b = 348.15
    T_b = T_b
    mf = 1
    n_wf = len(wf)
    n_beta = len(beta)
    rr_evap = np.zeros((n_wf, n_beta))
    wf_evap = np.zeros((n_wf, n_beta))
    W_least = np.zeros((n_wf, n_beta))
    W_least_with_recirc = np.zeros((n_wf, n_beta))

    for i,w in enumerate(wf):
        mv = rr*mf
        mb = (1-rr)*mf
        wb = w/(1-rr)
        for j,b in enumerate(beta):
            mb_evap = mb/(1-b)
            mb_rec = mb_evap-mb
            mf_evap = mb_rec + mf
            rr_evap[i,j] = mv/mf_evap
            wf_evap[i,j] = wb*mb_evap/mf_evap
            W_least[i,j] = get_W_least(rr,mf,mb,mv,w,wb,T_f,T_b)
            W_least_with_recirc[i,j] = get_W_least(rr_evap[i,j],mf_evap, mb_evap,mv,wf_evap[i,j],wb,T_f_recirc,T_b)

    print(W_least-W_least_with_recirc)
    fig = plt.figure()
    for j,b in enumerate(beta):
        plt.plot(wf*1e3, ((W_least_with_recirc[:,j]/W_least[:,j])-1)*100,label=str(b))
    plt.legend(frameon=False, title='Portion of brine\nrecirculated',fontsize=8)
    plt.xlim([min(wf*1e3), max(wf*1e3)])
    plt.xlabel('System feed concentration (g/kg)')
    plt.ylabel(r'$(W_{least,recirc} - W_{least})/W_{least}\times 100$ (%)')
    fig.set_size_inches(3.25,3.25)
    plt.title(r'Recovery = '+ str(rr*100) +' %')
    fig.tight_layout()
    plt.show()
    return

def get_W_least(r, mf,mb, md, wf, wb,T_f,T_b):# normalized to m_vapor
    t_f = T_f-273.15
    t_b = T_b-273.15
    t_v = t_b # C
    g_f = gibbs_sw(t_f,wf) # J/kg-K
    g_b = gibbs_sw(t_b, wb)  # J/kg-K
    g_d = gibbs_w(t_v)  # J/kg-K
    W_least = md*g_d + mb*g_b - mf*g_f
    # W_least = g_d - g_b + 1 / r * (g_b - g_f)
    return W_least/md*1e-3 # kJ/kg-K


def save_W_least():
    mf = 40 # kg/s
    t = 25
    T = t + 273.15
    wf = [25, 50, 75, 100, 125, 150]
    rr = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    results = np.empty((len(wf), len(rr)))
    for i, S in enumerate(wf):
        print('wf: ', S)
        for j, r in enumerate(rr):
            print('rr: ', r)
            wb = S/(1-r)
            mv = r*mf
            mb = mf-mv
            g_f = gibbs_sw(t,S) # J/kg-K
            g_b = gibbs_sw(t,wb) # J/kg-K
            g_d = gibbs_w(t) # J/kg-K
            # print('Gibbs flows: ', W_least)
            # print('Gibbs calc: ', W_least*1e-3)
            # g_f = h_sw(t,S)-T*s_sw(t,S)
            # g_b = h_sw(t,wb) -T*s_sw(t,wb)
            # g_d = h_w(t) - T*s_w(t)
            # W_least = (mb*g_b + mv*g_d - mf*g_f)
            W_least = g_d - g_b + 1/r*(g_b-g_f)
            results[i,j] = W_least*1e-3
            print(results[i,j])

    data = np.transpose(results)
    pd.DataFrame(data).to_csv('C:/Users/carso/Documents/MVC/watertap_results/W_least_g_direct_nayar_kJ_kg.csv', index=False)

def gibbs_w(t):
    c1 = 1.0677e2
    c2 = -1.4303
    c3 = -7.6139
    c4 = 8.3627e-3
    c5 = -7.8754e-6
    return c1 + c2 * t + c3 * t ** 2 + c4 * t ** 3 + c5 * t ** 4

def gibbs_sw(t,S):
    b1 = -2.4176e2
    b2 = -6.2462e-1
    b3 = 7.4761e-3
    b4 = 1.3836e-3
    b5 = -6.7157e-6
    b6 = 5.1993e-4
    b7 = 9.9176e-9
    b8 = 6.6448e1
    b9 = 2.0681e-1
    # return gibbs_w(t) + S*(b1 + b2*t + b3*t**2 + b4*S**2*t + b5*S**2*t**2 + b6*S**3 + b7*S**3*t**2 + b8*np.log(S) + b9*np.log(S)*t)
    return gibbs_w(t) + b1*S + b2*S*t + b3*S*t**2 + b4*S**2*t + b5*S**2*t**2 + b6*S**3 + b7*S**3*t**2 + b8*S*np.log(S) + b9*S*np.log(S)*t


def h_w(t):
    #t in C
    a0 = 141.355
    a1 = 4202.07
    a2 = -0.535
    a3 = 0.004
    return a0 + a1*t + a2*t**2 + a3*t**3

def h_sw(t,S_g):
    S = S_g/1e3
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
    term = b1 + b2*S + b3*S**2 + b4*S**3 + b5*t + b6*t**2 + b7*t**3 + b8*S*t + b9*S**2*t + b10*S*t**2

    return h_w(t) -S*term

def s_w(t):
    a0 = 0.1543
    a1 = 15.383
    a2 = -2.996e-2
    a3 = 8.193e-5
    a4 = -1.370e-7
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4

def s_sw(t,S_g):
    S = S_g/1e3
    b1 = -4.231e2
    b2 = 1.463e4
    b3 = -9.88e4
    b4 = 3.095e5
    b5 = 2.562e1
    b6 = -1.443e-1
    b7 = 5.879e-4
    b8 = -6.111e1
    b9 = 8.041e1
    b10 = 3.035e-1
    term = b1 + b2*S + b3*S**2 + b4*S**3 + b5*t + b6*t**2 + b7*t**3 + b8*S*t + b9*S**2*t + b10*S*t**2
    return s_w(t) - S*term


if __name__ == "__main__":
    m = main()
