import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np


def main():
    plot_rr_sys_rr_evap(wf=0.1,rr=np.arange(0.2,0.8,0.1), beta=[0,0.1,0.3,0.5,0.7])
    plot_wf_sys_wf_evap(wf=np.arange(0.025,0.15,0.025), rr=0.5,beta=[0,0.1,0.3,0.5,0.7])

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



if __name__ == "__main__":
    m = main()
