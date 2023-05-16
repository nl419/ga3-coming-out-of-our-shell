import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from entry_exit_coeffs import K_c_plus_K_e, F_1, F_2

# Geometry configuration
L_tube = 0.35
d_i = 0.006
d_o = 0.008
d_noz = 0.02
d_sh = 0.064

# Material properties
rho = 990.1
mu = 6.51e-4
Pr = 4.31
k_tube = 386
cp = 4179
k_w = 0.632
T_shell_in = 20
T_tube_in = 60  ### 2 == hot / tube, 1 == cold / shell
Y = 0.014

def flow_rate_shell(p):
    """Find the flow rate for a given pressure drop for the shell side compressor. p in Pa"""
    comp_flow_rate_shell = [0.7083, 0.6417, 0.5750, 0.5080, 0.4250, 0.3583, 0.3083, 0.2417, 0.1917, 0.1583]
    comp_p_rise_shell = [0.1310, 0.2017, 0.2750, 0.3417, 0.4038, 0.4503, 0.4856, 0.5352, 0.5717, 0.5876]
    return np.interp(p / 1e5, comp_p_rise_shell, comp_flow_rate_shell)

def flow_rate_tube(p):
    """Find the flow rate for a given pressure drop for the tube side compressor. p in Pa"""
    comp_flow_rate_tube = [0.4722, 0.4340, 0.3924, 0.3507, 0.3021, 0.2535, 0.1979, 0.1493, 0.1111, 0.0694]
    comp_p_rise_tube = [0.0538, 0.1192, 0.1727, 0.2270, 0.2814, 0.3366, 0.3907, 0.4456, 0.4791, 0.5115]
    return np.interp(p / 1e5, comp_p_rise_tube, comp_flow_rate_tube)

def find_H_mdots(N_tube, N_baffle, is_square = False):
    """Find overall heat transfer coefficient for a given number of tubes and number of baffles"""
    A_tube = (np.pi / 4) * d_i**2               # Area of a single tube
    A_shell = (np.pi / 4) * d_sh**2             # Area of the shell (ignoring baffles, tubes, etc)
    A_noz  = (np.pi / 4) * d_noz**2             # Area of inlet / outlet nozzles
    B = L_tube / (N_baffle + 1)                 # Shell baffle factor
    A_shell_flow = (d_sh / Y) * (Y - d_o) * B   # Flow area of shell taking into account baffles, tubes, etc
    sigma = (N_tube * A_tube) / A_shell 
    d_sh_characteristic = d_sh * (A_shell_flow / A_shell)
    A_i = np.pi*d_i*L_tube 
    A_o = np.pi*d_o*L_tube

    a = 0.2
    c = 0.2
    if is_square:
        a = 0.34
        c = 0.15

    # Start with initial guesses for both `mdot`s, iterate to find the intersection with compressor characteristic.
    mdot_tube = 0.5
    mdot_shell = 0.5
    mdot_tube_old = 0
    mdot_shell_old = 0
    tolerance = 1e-3
    relaxation_factor = 0.8

    while abs(mdot_tube - mdot_tube_old) > tolerance and abs(mdot_shell - mdot_shell_old) > tolerance:
        mdot_shell_old = mdot_shell
        mdot_tube_old = mdot_tube

        V_tube = mdot_tube / (rho * A_tube * N_tube)
        Re_tube = rho * V_tube * d_i / mu
        V_noz_tube = mdot_tube / (rho * A_noz)
        f = (1.82 * np.log10(Re_tube) - 1.64) ** (-2)

        del_p_ends_tube = 0.5 * rho * (V_tube**2) * K_c_plus_K_e(sigma, Re_tube)
        del_p_noz_tube = rho * V_noz_tube**2
        del_p_friction_tube = 0.5 * rho * (V_tube**2) * (f * L_tube / d_i)
        del_p_total_tube = del_p_ends_tube + del_p_noz_tube + del_p_friction_tube

        V_sh = mdot_shell / (rho * A_shell_flow)
        Re_sh = rho * V_sh * d_sh_characteristic / mu
        del_p_friction_shell = 4 * a * Re_sh**(-0.15) * N_tube * rho * V_sh**2
        V_noz_shell = mdot_shell / (rho * A_noz)
        del_p_noz_shell = rho * V_noz_shell**2
        del_p_total_shell = del_p_friction_shell + del_p_noz_shell

        mdot_shell = flow_rate_shell(del_p_total_shell) * relaxation_factor + mdot_shell * (1 - relaxation_factor)
        mdot_tube = flow_rate_tube(del_p_total_tube) * relaxation_factor + mdot_tube * (1 - relaxation_factor)

    Nu_i = 0.023 * Re_tube**0.8 * Pr**0.3
    Nu_o = c * Re_sh**0.6 * Pr**0.3

    h_i = Nu_i * k_w / d_i
    h_o = Nu_o * k_w / d_o

    H = 1 / ((1/h_i) + (A_i*np.log(d_o/d_i))/(2*np.pi*k_tube*L_tube) + ((1/h_o)*(A_i/A_o)))
    return H, mdot_shell, mdot_tube

def find_Q(N_tube, N_b, use_entu = False):
    H, mdot_shell, mdot_tube  = find_H_mdots(N_tube, N_b)
    C_min = cp * min(mdot_shell, mdot_tube)
    C_max = cp * max(mdot_shell, mdot_tube)
    R_c = C_min / C_max

    # Note: if T_tube_in - T_shell_out == T_tube_out - T_shell_in, there is a div/0 that breaks the iteration.
    # Also, if T_tube_in == T_shell_out or T_tube_out == T_shell_in, there is a log(0) or log(NaN)
    T_shell_out = 30
    T_tube_out = 40
    T_shell_out_old = 0
    T_tube_out_old = 0
    tolerance = 1e-3
    relaxation_factor = 0.8
    if use_entu:
        # ENTU tends to be more stable and tends to converge faster.
        # Typically 0.23 ms for ENTU and 0.3 ms for LMTD on my machine
        relaxation_factor = 1.0

    while abs(T_shell_out - T_shell_out_old) > tolerance and abs(T_tube_out - T_tube_out_old) > tolerance:
        T_shell_out_old = T_shell_out
        T_tube_out_old = T_tube_out

        R = (T_tube_in - T_tube_out) / (T_shell_out - T_shell_in)
        P = (T_shell_out - T_shell_in) / (T_tube_in - T_shell_in)
        F = F_1(P, R) # use F_1 for one pass, F_2 for 2 passes
        A = N_tube * np.pi * d_i * L_tube * F

        if use_entu:
            NTU = H * A / C_min
            effectiveness = (1 - np.exp(-NTU * (1 - R_c))) / (1 - R_c * np.exp(-NTU * (1 - R_c)))
            Q_max = C_min * (T_tube_in - T_shell_in)
            Q = effectiveness * Q_max 
        else:
            lmtd = ((T_tube_in - T_shell_out) - (T_tube_out - T_shell_in)) / (np.log((T_tube_in - T_shell_out) / (T_tube_out - T_shell_in)))
            Q = H * A * lmtd

        C_tube = mdot_tube * cp
        C_shell = mdot_shell * cp

        T_shell_out = (T_shell_in + Q / C_shell) * relaxation_factor + T_shell_out * (1 - relaxation_factor)
        T_tube_out = (T_tube_in - Q / C_tube) * relaxation_factor + T_tube_out * (1 - relaxation_factor)
    
    return Q


def many_configs():
    N_tubes_array = np.arange(10, 20, 1)
    N_baffles_array = np.arange(3, 8)
    Q_dots_ENTU = np.zeros((len(N_tubes_array), len(N_baffles_array)))
    Q_dots_LMTD = np.zeros((len(N_tubes_array), len(N_baffles_array)))
    for i,N_tubes in enumerate(N_tubes_array):
        for j,N_baffles in enumerate(N_baffles_array):
            Q_dots_ENTU[i,j] = find_Q(N_tubes, N_baffles, use_entu=True)
            Q_dots_LMTD[i,j] = find_Q(N_tubes, N_baffles, use_entu=False)

    print(Q_dots_ENTU)
    print(Q_dots_LMTD)
    
def one_config():
    N_tubes = 13
    N_baffles = 9
    print(find_Q(N_tubes, N_baffles, use_entu=True))
    print(find_Q(N_tubes, N_baffles, use_entu=False))

def benchmark():
    N_tubes = 13
    N_baffles = 9
    import timeit
    num_iters = 2000
    print(f'ENTU: {timeit.timeit("find_Q(N_tubes, N_baffles, use_entu=True)",  globals=locals(), setup="from __main__ import find_Q", number=num_iters) * 1e3 / num_iters} ms')
    print(f'LMTD: {timeit.timeit("find_Q(N_tubes, N_baffles, use_entu=False)", globals=locals(), setup="from __main__ import find_Q", number=num_iters) * 1e3 / num_iters} ms')

if __name__ == "__main__":
    # one_config()
    benchmark()
    
