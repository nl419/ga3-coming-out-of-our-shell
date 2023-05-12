import numpy as np

# Config
L_tube = 0.35
d_i = 0.006
d_o = 0.008

# Properties
rho = 990.1
mu = 6.51e-4
Pr = 4.31
k_tube = 386
cp = 4179
k_w = 0.632
T1_in = 20
T2_in = 60

# Pressure drop for given config
## Pressure drop for tube side
V_tube = 1
V_noz_2 = 2
Re_tube = 1000
f = (1.82 * np.log10(Re_tube) - 1.64) ** (-2)
K_e = 0.8
K_c = 0.45
del_p_tube = 0.5 * rho * V_tube ** 2 * (f * L_tube / d_i + K_e + K_c) \
    + 0.5 * rho * V_noz_2 ** 2

## Pressure drop for shell side
d_sh = 0.2 #for now
Y = 0.014 #for now
N = 13 #for now
N_b = 9 #for now
B = L_tube/(N_b + 1)
A_sh = (d_sh/Y)*(Y - d_o)*B

V_sh = 2 #for now
Re_sh = rho*V_sh*L_tube/mu

A_pipe =(np.pi/4)*(d_sh) 
d_sh_1 = d_sh * (A_sh/A_pipe) #propose characteristic length
a = 0.2 #this is a constant, 0.2 for triangular arrangement
Re_sh = rho*V_sh*d_sh_1/mu
del_p_sh = 4*a*Re_sh**(-0.15)*N*rho*V_sh**2

V_noz_1 = 1.6 #for now
del_p_noz_1 = rho*V_noz_1**2

del_p_1 = del_p_sh + del_p_noz_1

# Heat transfer coeff
A_i = np.pi*d_i*L_tube
A_o = np.pi*d_o*L_tube
c = 0.2 #constant, 0.2 for tringular, 0.15 for square
Nu_i = 0.023*(Re_tube**(0.8))*(Pr**(0.3))
Nu_o = c*(Re_sh**0.6)*(Pr**0.3)

h_i = Nu_i*k_w/d_i
h_o = Nu_o*k_w/d_o

H = ((1/h_i) + ((A_i*np.log(d_o/d_i))/(2*np.pi*k_tube*L_tube) + ((1/h_o)*(A_i/A_o))))**(-1)

# Q
T1_out = 30 #for now, need to iterate in order to find the exact temperatures
T2_out = 50
del_T_lm = ((T2_in - T1_out) - (T2_out - T1_in))/(np.log((T2_in - T1_out)/(T2_out - T1_in)))
F = 0.7 #correction factor, dependent on number of tube passes, will need to compute exact value
A = N*np.pi*d_i*L_tube
Q_dot = H*A*del_T_lm*F