import numpy as np
from matplotlib import pyplot as plt
from entry_exit_coeffs import K_c_plus_K_e, F_1, F_2
from mass_estimate import calculate_tube_length_baffle_spacing
from dataclasses import dataclass

# Geometry configuration
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

@dataclass
class HXGeometry:
    N_tube: int
    N_baffle: int
    L_tube: float
    baffle_spacing: float
    shell_passes: int = 1
    tube_passes: int = 1

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

def find_H_mdots(geom: HXGeometry, is_square = False):
    """Find overall heat transfer coefficient for a given number of tubes and number of baffles"""
    A_tube_inner = (np.pi / 4) * d_i**2               # Area of a single tube
    A_tube_outer = (np.pi / 4) * d_o**2               # Area of a single tube
    A_shell = (np.pi / 4) * d_sh**2             # Area of the shell (ignoring baffles, tubes, etc)
    A_noz  = (np.pi / 4) * d_noz**2             # Area of inlet / outlet nozzles
    sigma = (geom.N_tube * A_tube_outer) / A_shell 
    baffle_spacing = geom.L_tube / (geom.N_baffle + 1)                 # Shell baffle factor
    A_shell_flow = d_sh * baffle_spacing * sigma   # Flow area of shell taking into account baffles, etc
    d_sh_characteristic = d_sh * (A_shell_flow / A_shell)
    # d_sh_characteristic = d_sh * (1 - sigma)
    A_i = np.pi*d_i*geom.L_tube 
    A_o = np.pi*d_o*geom.L_tube

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

        V_tube = mdot_tube / (rho * A_tube_inner * geom.N_tube)
        Re_tube = rho * V_tube * d_i / mu
        V_noz_tube = mdot_tube / (rho * A_noz)
        f = (1.82 * np.log10(Re_tube) - 1.64) ** (-2)

        del_p_ends_tube = 0.5 * rho * (V_tube**2) * K_c_plus_K_e(sigma, Re_tube)
        del_p_noz_tube = rho * V_noz_tube**2
        del_p_friction_tube = 0.5 * rho * (V_tube**2) * (f * geom.L_tube / d_i)
        del_p_total_tube = del_p_ends_tube + del_p_noz_tube + del_p_friction_tube

        V_sh = mdot_shell / (rho * A_shell_flow)
        Re_sh = rho * V_sh * d_sh_characteristic / mu
        del_p_friction_shell = 4 * a * Re_sh**(-0.15) * geom.N_tube * rho * V_sh**2
        V_noz_shell = mdot_shell / (rho * A_noz)
        del_p_noz_shell = rho * V_noz_shell**2
        del_p_total_shell = del_p_friction_shell + del_p_noz_shell

        mdot_shell = flow_rate_shell(del_p_total_shell) * relaxation_factor + mdot_shell * (1 - relaxation_factor)
        mdot_tube = flow_rate_tube(del_p_total_tube) * relaxation_factor + mdot_tube * (1 - relaxation_factor)

    Nu_i = 0.023 * Re_tube**0.8 * Pr**0.3
    Nu_o = c * Re_sh**0.6 * Pr**0.3

    h_i = Nu_i * k_w / d_i
    h_o = Nu_o * k_w / d_o

    H = 1 / ((1/h_i) + (A_i*np.log(d_o/d_i))/(2*np.pi*k_tube*geom.L_tube) + ((1/h_o)*(A_i/A_o)))
    return H, mdot_shell, mdot_tube

def find_Q(geom: HXGeometry, use_entu = False):
    H, mdot_shell, mdot_tube  = find_H_mdots(geom)
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
        A = geom.N_tube * np.pi * d_i * geom.L_tube * F

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

def benchmark():
    N_tubes = 13
    N_baffles = 9
    L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, N_tubes, N_baffles)
    geom = HXGeometry(N_tubes, N_baffles, L_tube, baffle_spacing)

    import timeit
    num_iters = 2000
    print(f'ENTU: {timeit.timeit("find_Q(geom, use_entu=True)",  globals=locals(), setup="from __main__ import find_Q", number=num_iters) * 1e3 / num_iters} ms')
    print(f'LMTD: {timeit.timeit("find_Q(geom, use_entu=False)", globals=locals(), setup="from __main__ import find_Q", number=num_iters) * 1e3 / num_iters} ms')

def brute_force():
    max_q = 0
    max_n_tubes = 0
    max_n_baffles = 0
    for N_tubes in np.arange(3, 30):
        for N_baffles in np.arange(2, 30):
            L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, N_tubes, N_baffles)
            geom = HXGeometry(N_tubes, N_baffles, L_tube, baffle_spacing)
            q = find_Q(geom, use_entu=True)
            if q > max_q:
                max_q = q
                max_n_baffles = N_baffles
                max_n_tubes = N_tubes
    print(f"max_q = {max_q}, max_n_baffles = {max_n_baffles}, max_n_tubes = {max_n_tubes}")

def plot_graphs():
    max_n_tubes = 10
    max_n_baffles = 9
    n_tubes_array = np.arange(max_n_tubes - 4, max_n_tubes + 4)
    n_baffles_array = np.arange(max_n_baffles - 4, max_n_baffles + 4)

    qs = np.zeros(np.shape(n_tubes_array))
    ls = np.zeros(np.shape(n_tubes_array))

    for i,n_tubes in enumerate(n_tubes_array):
        L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, n_tubes, max_n_baffles)
        geom = HXGeometry(n_tubes, max_n_baffles, L_tube, baffle_spacing)
        qs[i] = find_Q(geom, use_entu=True)
        ls[i] = L_tube
    
    plt.plot(n_tubes_array, qs)
    plt.show()
    plt.plot(n_tubes_array, ls)
    plt.show()

    qs = np.zeros(np.shape(n_baffles_array))
    ls = np.zeros(np.shape(n_baffles_array))

    for i,n_baffles in enumerate(n_baffles_array):
        L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, max_n_tubes, n_baffles)
        geom = HXGeometry(max_n_tubes, n_baffles, L_tube, baffle_spacing)
        qs[i] = find_Q(geom, use_entu=True)
        ls[i] = L_tube
    
    plt.plot(n_baffles_array, qs, label="Varying baffles")
    plt.show()
    plt.plot(n_baffles_array, ls)
    plt.show()

if __name__ == "__main__":
    # one_config()
    # benchmark()
    # brute_force()
    plot_graphs()
