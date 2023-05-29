import numpy as np
from matplotlib import pyplot as plt
from entry_exit_coeffs import K_c_plus_K_e
from mass_estimate import calculate_tube_length_baffle_spacing
from dataclasses import dataclass
from correction_factor import correction_factor
from scipy.optimize import minimize

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
Y = 0.012

@dataclass
class HXGeometry:
    N_tube: int
    N_baffle: int
    L_tube: float
    baffle_spacing: float
    shell_passes: int = 1
    tube_passes: int = 1

@dataclass
class PastResults:
    geom: HXGeometry
    dp_cold: float
    dp_hot: float
    q: float

all_past_results = {
    2022: [
        PastResults(HXGeometry(14, 12, 0.224, 11.45e-3, 1, 1), 0.305e5, 0.122e5, 11.53e3),
        PastResults(HXGeometry(16,  6, 0.210, 50.00e-3, 1, 1), 0.160e5, 0.118e5, 11.66e3),
        PastResults(HXGeometry(20,  4, 0.162, 34.00e-3, 1, 2), 0.179e5, 0.190e5, 13.11e3),
        PastResults(HXGeometry(12,  8, 0.256, 19.50e-3, 1, 2), 0.187e5, 0.221e5, 13.45e3),
        PastResults(HXGeometry(24,  4, 0.135, 18.16e-3, 1, 4), 0.216e5, 0.206e5, 13.37e3),
        PastResults(HXGeometry(20,  6, 0.163, 18.00e-3, 2, 4), 0.276e5, 0.299e5, 13.54e3)
    ],
    2018: [
        PastResults(HXGeometry(20,  7, 0.132, 23.80e-3, 2, 4), 0.202e5, 0.292e5, 10.52e3),
        PastResults(HXGeometry(24,  7, 0.121, 22.00e-3, 2, 4), 0.241e5, 0.179e5, 11.74e3),
        PastResults(HXGeometry(16,  5, 0.121, 43.80e-3, 2, 2), 0.244e5, 0.179e5, 11.50e3),
    ],
    2019: [
        PastResults(HXGeometry(16, 6, 0.2127, 12.8e-3, 2, 4), 0.437e5, 0.391e5, 12.78e3),
        PastResults(HXGeometry(14, 11, 0.237, 14.5e-3, 2, 2), 0.458e5, 0.219e5, 12.82e3),
        PastResults(HXGeometry(16, 6, 0.213, 25.0e-3, 1, 2), 0.275e5, 0.212e5, 14.23e3),
        PastResults(HXGeometry(20, 6, 0.184, 22.5e-3, 1, 2), 0.362e5, 0.348e5, 16.27e3),
        PastResults(HXGeometry(18, 8, 0.203, 16.75e-3, 1, 2), 0.325e5, 0.200e5, 17.24e3),
    ]
}

def flow_rate_shell(p, year):
    """Find the flow rate for a given pressure drop for the shell side compressor. p in Pa"""
    all_flow_rates_p_rise_shell = {
        2023: (
            (0.7083, 0.6417, 0.5750, 0.5080, 0.4250, 0.3583, 0.3083, 0.2417, 0.1917, 0.1583),
            (0.1310, 0.2017, 0.2750, 0.3417, 0.4038, 0.4503, 0.4856, 0.5352, 0.5717, 0.5876)),
        2022: (
            (0.5833, 0.5083, 0.4750, 0.4250, 0.3792, 0.3417, 0.2958, 0.2583, 0.2125, 0.1708),
            (0.1113, 0.2157, 0.2538, 0.3168, 0.3613, 0.4031, 0.4511, 0.4846, 0.5181, 0.5573)),
        2019: (
            (0.6917, 0.6750, 0.6292, 0.5917, 0.5458, 0.5083, 0.4625, 0.4250, 0.3792, 0.3417, 0.2958, 0.2542, 0.2125, 0.1708),
            (0.1475, 0.1619, 0.2178, 0.2607, 0.3041, 0.3417, 0.3756, 0.4118, 0.4423, 0.4711, 0.5031, 0.5297, 0.5561, 0.5823)),
        2018: (
            (0.4426, 0.4255, 0.4055, 0.3913, 0.3799, 0.3628, 0.3485, 0.3286, 0.3058, 0.2801, 0.2573, 0.2317, 0.2060, 0.1861, 0.1576, 0.1319, 0.1034, 0.0806, 0.0664, 0.0521),
            (0.1068, 0.1418, 0.1779, 0.2056, 0.2382, 0.2601, 0.2858, 0.3187, 0.3627, 0.4037, 0.4426, 0.4845, 0.5213, 0.5569, 0.6036, 0.6412, 0.6838, 0.7121, 0.7343, 0.7744))
    }

    comp_flow_rate_shell, comp_p_rise_shell = all_flow_rates_p_rise_shell[year]

    return np.interp(p / 1e5, comp_p_rise_shell, comp_flow_rate_shell)

def flow_rate_tube(p, year):
    """Find the flow rate for a given pressure drop for the tube side compressor. p in Pa"""
    all_flow_rates_p_rise_shell = {
        2023: (
            (0.4722, 0.4340, 0.3924, 0.3507, 0.3021, 0.2535, 0.1979, 0.1493, 0.1111, 0.0694),
            (0.0538, 0.1192, 0.1727, 0.2270, 0.2814, 0.3366, 0.3907, 0.4456, 0.4791, 0.5115)),
        2022: (
            (0.4583, 0.4236, 0.4010, 0.3611, 0.3125, 0.2639, 0.2222, 0.1597, 0.1181, 0.0694),
            (0.1333, 0.1756, 0.2024, 0.2577, 0.3171, 0.3633, 0.4233, 0.4784, 0.5330, 0.5715)),
        2019: (
            (0.5382, 0.5278, 0.4931, 0.4549, 0.4201, 0.3854, 0.3507, 0.3160, 0.2813, 0.2465, 0.2118, 0.1771, 0.1424, 0.1076, 0.0694),
            (0.1101, 0.1315, 0.1800, 0.2185, 0.2537, 0.2999, 0.3440, 0.3780, 0.4149, 0.4547, 0.5005, 0.5271, 0.5677, 0.5971, 0.6045)),
        2018: (
            (0.4954, 0.4805, 0.4640, 0.4475, 0.4310, 0.4145, 0.3980, 0.3815, 0.3650, 0.3485, 0.3320, 0.3155, 0.2990, 0.2825, 0.2660, 0.2495, 0.2330, 0.2165, 0.2000, 0.1819, 0.1670, 0.1472, 0.1307, 0.1142, 0.1010, 0.0845, 0.0680, 0.0515),
            (0.0989, 0.1245, 0.1541, 0.1827, 0.2083, 0.2339, 0.2625, 0.2880, 0.3115, 0.3330, 0.3575, 0.3800, 0.4014, 0.4249, 0.4503, 0.4647, 0.4900, 0.5134, 0.5337, 0.5470, 0.5703, 0.5966, 0.6068, 0.6150, 0.6242, 0.6304, 0.6375, 0.6457))
    }

    comp_flow_rate_tube, comp_p_rise_tube = all_flow_rates_p_rise_shell[year]

    return np.interp(p / 1e5, comp_p_rise_tube, comp_flow_rate_tube)

def find_H_mdots(geom: HXGeometry, is_square = False, fix_mdots = False, mdots = [0,0], new_ho = False,
                 fac1 = 1.0, fac2 = 1.0, fac3 = 1.0, fac4 = 1.0, year = 2023):
    """Find overall heat transfer coefficient for a given number of tubes and number of baffles"""
    #assert(geom.N_tube % geom.tube_passes == 0)

    A_tube_inner = (np.pi / 4) * d_i**2               # Area of a single tube
    A_tube_outer = (np.pi / 4) * d_o**2               # Area of a single tube
    A_shell = (np.pi / 4) * d_sh**2             # Area of the shell (ignoring baffles, tubes, etc)
    A_noz  = (np.pi / 4) * d_noz**2             # Area of inlet / outlet nozzles
    sigma = (geom.N_tube * A_tube_outer) / A_shell 
    baffle_spacing = geom.L_tube / (geom.N_baffle + 1)                 # Shell baffle factor
    A_shell_flow = d_sh * baffle_spacing * (1 - sigma) / geom.shell_passes   # Flow area of shell taking into account baffles, etc
    # d_sh_characteristic = d_sh * (A_shell_flow / A_shell)
    d_sh_characteristic = d_sh * (1 - sigma)
    A_i = np.pi*d_i*geom.L_tube 
    A_o = np.pi*d_o*geom.L_tube

    a = 0.2
    c = 0.2
    if is_square:
        a = 0.34
        c = 0.15

    if not fix_mdots:
        # Start with initial guesses for both `mdot`s, iterate to find the intersection with compressor characteristic.
        mdot_tube = 0.5
        mdot_shell = 0.5
        mdot_tube_old = 0
        mdot_shell_old = 0
        tolerance = 1e-3
        relaxation_factor = 0.1

        while abs(mdot_tube - mdot_tube_old) > tolerance or abs(mdot_shell - mdot_shell_old) > tolerance:
            mdot_shell_old = mdot_shell
            mdot_tube_old = mdot_tube

            V_tube = mdot_tube / (rho * A_tube_inner * geom.N_tube / geom.tube_passes)
            Re_tube = rho * V_tube * d_i / mu
            V_noz_tube = mdot_tube / (rho * A_noz)
            f = (1.82 * np.log10(Re_tube) - 1.64) ** (-2)

            del_p_ends_tube = 0.5 * rho * (V_tube**2) * K_c_plus_K_e(sigma, Re_tube) * geom.tube_passes
            del_p_noz_tube = rho * V_noz_tube**2
            del_p_friction_tube = 0.5 * rho * (V_tube**2) * (f * geom.L_tube * geom.tube_passes / d_i)
            del_p_total_tube = del_p_ends_tube + del_p_noz_tube + del_p_friction_tube

            V_sh = mdot_shell / (rho * A_shell_flow)
            Re_sh = rho * V_sh * d_sh_characteristic / mu
            del_p_friction_shell = 4 * a * Re_sh**(-0.15) * geom.N_tube * rho * V_sh**2
            V_noz_shell = mdot_shell / (rho * A_noz)
            del_p_noz_shell = rho * V_noz_shell**2
            del_p_total_shell = 12.1637 * del_p_friction_shell * geom.shell_passes + del_p_noz_shell

            mdot_shell = flow_rate_shell(del_p_total_shell, year) * relaxation_factor * 1000 / rho + mdot_shell * (1 - relaxation_factor)
            mdot_tube = flow_rate_tube(del_p_total_tube, year) * relaxation_factor * 1000 / rho + mdot_tube * (1 - relaxation_factor)

    else:

        mdot_shell = mdots[0]
        mdot_tube = mdots[1]

        V_tube = mdot_tube / (rho * A_tube_inner * geom.N_tube / geom.tube_passes)
        Re_tube = rho * V_tube * d_i / mu

        V_sh = mdot_shell / (rho * A_shell_flow)
        Re_sh = rho * V_sh * d_sh_characteristic / mu

    Nu_i = 0.023 * Re_tube**0.8 * Pr**0.3
    h_i = Nu_i * k_w / d_i

    if new_ho:

        if is_square:
            
            free_area = Y**2 - (np.pi/4) * d_o**2 # Free area looking along the axis of the tubes
            wetted_perimeter = np.pi * d_o

        else:

            free_area = (3)**0.5 / 4 * Y**2 - (np.pi/8) * d_o**2
            wetted_perimeter = 0.5 * np.pi * d_o

        De = 4 * free_area / wetted_perimeter
        A_shell_flow = d_sh * baffle_spacing * (Y - d_o) / (Y * 144)
        #A_shell_flow = d_sh * baffle_spacing * (1 - sigma) / (geom.shell_passes)
        Gs = mdot_shell / A_shell_flow
        

        h_o = 0.36 * (De * Gs / mu)**0.55 * (cp * mu / k_w)**(1/3) * k_w / De
        
    else:

        Nu_o = c * Re_sh**0.6 * Pr**0.3
        h_o = Nu_o * k_w / d_o

    H = 1 / ((1/h_i) + (A_i*np.log(d_o/d_i))/(2*np.pi*k_tube*geom.L_tube) + ((1/h_o)*(A_i/A_o)))
    return H, mdot_shell, mdot_tube

def find_Q(geom: HXGeometry, use_entu = False, fix_mdots = False, mdots = [0,0], new_ho = False):
    H, mdot_shell, mdot_tube  = find_H_mdots(geom, fix_mdots=fix_mdots, mdots=mdots, new_ho=new_ho)
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

    while abs(T_shell_out - T_shell_out_old) > tolerance or abs(T_tube_out - T_tube_out_old) > tolerance:
        T_shell_out_old = T_shell_out
        T_tube_out_old = T_tube_out

        F = correction_factor(T_shell_in, T_tube_in, T_shell_out, T_tube_out, geom.shell_passes)
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

def brute_force_11():
    max_q = 0
    max_n_tubes = 0
    max_n_baffles = 0
    for N_tubes in np.arange(3, 22):
        for N_baffles in np.arange(2, 30):
            L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, N_tubes, N_baffles)
            geom = HXGeometry(N_tubes, N_baffles, L_tube, baffle_spacing)
            q = find_Q(geom, use_entu=True)
            if q > max_q:
                max_q = q
                max_n_baffles = N_baffles
                max_n_tubes = N_tubes
    print(f"max_q = {max_q}, max_n_baffles = {max_n_baffles}, max_n_tubes = {max_n_tubes}")

def brute_force_12():
    max_q = 0
    max_n_tubes = 0
    max_n_baffles = 0
    for N_tubes in np.arange(2, 11) * 2:
        for N_baffles in np.arange(2, 10):
            # print(N_tubes, N_baffles)
            L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, N_tubes, N_baffles)
            geom = HXGeometry(N_tubes, N_baffles, L_tube, baffle_spacing, tube_passes=2)
            q = find_Q(geom, use_entu=True)
            if q > max_q:
                max_q = q
                max_n_baffles = N_baffles
                max_n_tubes = N_tubes
    print(f"max_q = {max_q}, max_n_baffles = {max_n_baffles}, max_n_tubes = {max_n_tubes}")

def brute_force_14():
    max_q = 0
    max_n_tubes = 0
    max_n_baffles = 0
    for N_tubes in np.arange(2, 6) * 4:
        for N_baffles in np.arange(2, 10):
            # print(N_tubes, N_baffles)
            L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, N_tubes, N_baffles)
            geom = HXGeometry(N_tubes, N_baffles, L_tube, baffle_spacing, tube_passes=4)
            q = find_Q(geom, use_entu=True)
            if q > max_q:
                max_q = q
                max_n_baffles = N_baffles
                max_n_tubes = N_tubes
    print(f"max_q = {max_q}, max_n_baffles = {max_n_baffles}, max_n_tubes = {max_n_tubes}")

def brute_force_custom():
    max_q = 0
    max_n_tubes = 0
    max_n_baffles = 0
    shell_passes = 1
    tube_passes = 6
    for N_tubes in np.arange(2, 14) * tube_passes:
        for N_baffles in np.arange(2, 15):
            L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(shell_passes, tube_passes, N_tubes, N_baffles)
            geom = HXGeometry(N_tubes, N_baffles, L_tube, baffle_spacing, tube_passes=tube_passes, shell_passes=shell_passes)
            q = find_Q(geom, use_entu=True)
            if q > max_q:
                max_q = q
                max_n_baffles = N_baffles
                max_n_tubes = N_tubes
            print(N_tubes, N_baffles, q)
    print(f"max_q = {max_q}, max_n_baffles = {max_n_baffles}, max_n_tubes = {max_n_tubes}")

def brute_force_all():
    for shell_passes in [1,2,3,4]:
        for tube_passes in [1,2,4,6,8]:
            max_q = 0
            max_n_tubes = 0
            max_n_baffles = 0
            for N_tubes in np.arange(1, 20):
                if N_tubes > 18:
                    break
                for N_baffles in np.arange(2, 15):
                    # print(N_tubes, N_baffles)
                    L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(shell_passes, tube_passes, N_tubes, N_baffles)
                    geom = HXGeometry(N_tubes, N_baffles, L_tube, baffle_spacing, tube_passes=tube_passes, shell_passes=shell_passes)
                    q = find_Q(geom, use_entu=False, new_ho = False)
                    if q > max_q:
                        max_q = q
                        max_n_baffles = N_baffles
                        max_n_tubes = N_tubes
            print(f"shell_passes = {shell_passes}, tube_passes = {tube_passes}, max_q = {max_q}, max_n_baffles = {max_n_baffles}, max_n_tubes = {max_n_tubes}")

def plot_graphs():
    tube_passes = 2
    max_n_tubes = 12
    max_n_baffles = 6
    n_tubes_array = max_n_tubes + np.arange(-3, 10) * tube_passes
    n_baffles_array = np.arange(max_n_baffles - 4, max_n_baffles + 4)

    qs = np.zeros(np.shape(n_tubes_array))
    ls = np.zeros(np.shape(n_tubes_array))
    areas = np.zeros(np.shape(n_tubes_array))

    for i,n_tubes in enumerate(n_tubes_array):
        L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, n_tubes, max_n_baffles)
        geom = HXGeometry(n_tubes, max_n_baffles, L_tube, baffle_spacing, tube_passes=tube_passes)
        qs[i] = find_Q(geom, use_entu=True)
        ls[i] = L_tube
        areas[i] = L_tube * n_tubes
    
    plt.plot(n_tubes_array, qs)
    plt.xlabel('number of tubes')
    plt.ylabel('Heat Transfer Q / W')
    plt.show()
    plt.plot(n_tubes_array, ls)
    plt.xlabel('number of tubes')
    plt.ylabel('Overall length / m')
    plt.show()
    plt.plot(n_tubes_array, areas)
    plt.xlabel('number of tubes')
    plt.ylabel('Surface area available for heat transfer / m2')
    plt.show()

    qs = np.zeros(np.shape(n_baffles_array))
    ls = np.zeros(np.shape(n_baffles_array))

    for i,n_baffles in enumerate(n_baffles_array):
        L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, max_n_tubes, n_baffles)
        geom = HXGeometry(max_n_tubes, n_baffles, L_tube, baffle_spacing, tube_passes=tube_passes)
        qs[i] = find_Q(geom, use_entu=True)
        ls[i] = L_tube
    
    plt.plot(n_baffles_array, qs, label="Varying baffles")
    plt.xlabel('number of baffles')
    plt.ylabel('Heat transfer Q / W')
    plt.show()
    plt.plot(n_baffles_array, ls)
    plt.xlabel('number of baffles')
    plt.ylabel('Overall length / m')
    plt.show()

def two_configs():
    tube_passes = 2
    n_tubes = 12
    n_baffles = 2
    L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, n_tubes, n_baffles)
    geom = HXGeometry(n_tubes, n_baffles, L_tube, baffle_spacing, tube_passes=tube_passes)
    find_Q(geom, use_entu=True)
    n_tubes = 20
    L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, n_tubes, n_baffles)
    geom = HXGeometry(n_tubes, n_baffles, L_tube, baffle_spacing, tube_passes=tube_passes)
    find_Q(geom, use_entu=True)

def one_config():
    tube_passes = 4
    shell_passes = 2
    n_tubes = 18
    n_baffles = 10
    # L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, n_tubes, n_baffles)
    L_tube = 0.25
    baffle_spacing = 0.02
    geom = HXGeometry(n_tubes, n_baffles, L_tube, baffle_spacing, tube_passes=tube_passes, shell_passes=shell_passes)
    print(find_Q(geom, use_entu=True))

def enforce_mass_flows():
    tube_passes = 2
    shell_passes = 1
    n_tubes = 18
    n_baffles = 8
    # L_tube, baffle_spacing = calculate_tube_length_baffle_spacing(1, 1, n_tubes, n_baffles)
    L_tube = 0.212
    baffle_spacing = 0.01675
    geom = HXGeometry(n_tubes, n_baffles, L_tube, baffle_spacing, tube_passes=tube_passes, shell_passes=shell_passes)
    print(find_Q(geom, use_entu=True, fix_mdots=True, mdots = [0.608*rho/1000, 0.483*rho/1000], new_ho=False))
    # mdots are [mdot_shell, mdot_tube]


def dp_shell_error(result_year: tuple, fac1: float, fac2: float, fac3: float, fac4: float, fac5: float):
    result: PastResults = result_year[0]
    geom = result.geom
    year = result_year[1]
    A_tube_outer = (np.pi / 4) * d_o**2               # Area of a single tube
    A_shell = (np.pi / 4) * d_sh**2             # Area of the shell (ignoring baffles, tubes, etc)
    A_noz  = (np.pi / 4) * d_noz**2             # Area of inlet / outlet nozzles
    sigma = (geom.N_tube * A_tube_outer) / A_shell 
    baffle_spacing = geom.L_tube / (geom.N_baffle + 1)                 # Shell baffle factor
    A_shell_flow = d_sh * baffle_spacing * (1 - sigma) / geom.shell_passes   # Flow area of shell taking into account baffles, etc
    d_sh_characteristic = d_sh * (1 - sigma)

    L_sh_eff = (geom.L_tube + d_sh * geom.N_baffle) * geom.shell_passes

    a = 0.2
    # Start with initial guesses for both `mdot`s, iterate to find the intersection with compressor characteristic.
    mdot_shell = 0.5
    mdot_shell_comp = 0
    tolerance = 1e-4
    relaxation_factor = 0.1

    while abs(mdot_shell - mdot_shell_comp) > tolerance:
        V_sh = mdot_shell / (rho * A_shell_flow)
        Re_sh = rho * V_sh * d_sh_characteristic / mu
        del_p_friction_shell = 4 * a * Re_sh**(-0.15) * geom.N_tube * rho * V_sh**2
        V_noz_shell = mdot_shell / (rho * A_noz)
        del_p_noz_shell = rho * V_noz_shell**2
        # del_p_total_shell = del_p_friction_shell * fac1 * (geom.shell_passes ** fac2) + (del_p_noz_shell * fac4) + geom.N_baffle * fac3 * V_sh**2
        # del_p_total_shell = del_p_friction_shell * fac1 * (L_sh_eff ** fac2) + del_p_noz_shell
        #del_p_total_shell = del_p_friction_shell * fac1 + del_p_noz_shell + V_sh**2 * geom.shell_passes * fac2 * 100

        del_p_total_shell = rho*V_sh**2*fac1 + geom.N_baffle*geom.shell_passes*V_sh**2*rho*fac2 + (geom.shell_passes-1)*rho*V_sh**2*fac3 + Re_sh**(-0.15)*geom.N_tube*rho*V_sh**2*fac5*geom.N_baffle
        
        mdot_shell_comp = flow_rate_shell(del_p_total_shell, year) * 1000 / rho

        mdot_shell = mdot_shell_comp * relaxation_factor + mdot_shell * (1 - relaxation_factor)
    return del_p_total_shell - result.dp_cold

def optimise_dp_coeffs():
    result_years = []
    for key,result_list in all_past_results.items():
        for result in result_list:
            result_years.append((result, key))

    def total_dp_shell_error(args):
        fac1, fac2, fac3, fac4, fac5 = args
        total = 0
        for result_year in result_years:
            err = dp_shell_error(result_year, fac1, fac2, fac3, fac4, fac5)
            total += err * err
        total = np.sqrt(total)
        print(args)
        print(total)
        # print()
        return total
    
    # print(total_dp_shell_error([1,1,1,1]))

    res = minimize(total_dp_shell_error, [1, 1, 1, 0.15, 0.8])
    print(res) # [1.45574161e-05, 1.23121351e+00, 5.64149288e+00, 9.92159081e+00]

def plot_dp_shell():
    # Get array of past dps
    # Get array of current dps
    # Plot as i,y-x

    result_years = []
    for key,result_list in all_past_results.items():
        for result in result_list:
            result_years.append((result, key))

    print(result_years)

    # args = [1.45574161e-05, 1.23121351e+00, 5.64149288e+00, 9.92159081e+00]
    args = [ 1.94779271e+02,  7.39740878e+00, -1.41953981e+02,  1.50000000e-01, -4.62675568e+00]
    fac1, fac2, fac3, fac4, fac5 = args

    errs = []

    for result_year in result_years:
        err = dp_shell_error(result_year, fac1, fac2, fac3, fac4, fac5)
        errs.append(err / result_year[0].dp_cold)

    plt.plot(errs, "o")
    plt.show() 

if __name__ == "__main__":
    # plot_graphs()
    # one_config()
    # benchmark()
    # brute_force_11()
    # brute_force_12()
    # brute_force_14()
    # brute_force_custom()
    # plot_graphs()
    # two_configs()
    brute_force_all()
    # enforce_mass_flows()
    # optimise_dp_coeffs()
    # plot_dp_shell()
