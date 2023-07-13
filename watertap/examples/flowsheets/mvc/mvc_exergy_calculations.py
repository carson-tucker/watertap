import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():


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
    main()