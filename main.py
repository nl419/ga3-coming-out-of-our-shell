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

# Pressure drop for given config
## Pressure drop for tube side
V_tube = 1
V_noz = 2
Re = 1000
f = (1.82 * np.log10(Re) - 1.64) ** (-2)
K_e = 0.8
K_c = 0.45
del_p_tube = 0.5 * rho * V_tube ** 2 * (f * L_tube / d_i + K_e + K_c) \
    + 0.5 * rho * V_noz ** 2

## Pressure drop for shell side
D_sh = 0.2 #for now
Y = 0.014 #for now
N = 13 #for now
N_b = 9 #for now
d_o = 0.008
B = L_tube/(N_b + 1)
A_sh = (D_sh/Y)*(Y - d_o)*B

V_sh = 2 #for now
Re_sh = 
# Heat transfer coeff

# Q