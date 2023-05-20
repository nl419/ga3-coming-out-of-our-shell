import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Material properties
rho = 990.1
mu = 6.51e-4
Pr = 4.31
k_tube = 386
cp = 4179
k_w = 0.632
T_shell_in = 20
T_tube_in = 60  ### 2 == hot / tube, 1 == cold / shell

@dataclass
class ConcentricGeometry:
    N_tube: int # Number of tubes PER DIRECTION
    L_tube: float
    d_i: float # Diameter of innermost wetted wall 
    d_m: float # Diameter of middle wall
    d_o: float # Diameter of outermost wetted wall
    passes: int = 1
    d_noz: float = 0.02

# For now, assume 0.6 mdot cold, 0.4 mdot hot
mdot_o = 0.7
mdot_i = 0.47

d_i = 6e-3
d_m = 8e-3
d_o = 10e-3

AX_i = (np.pi / 4) * d_i**2 # Cross-sectional area of inner tube
AX_o = (np.pi / 4) * (d_o**2 - d_m**2)  # Cross sectional area of outer annulus

# d_oh = 4 * AX_o / (np.pi * (d_o + d_m)) # Hydraulic diameter of outer annular area
d_oh = 64e-3 # Hydraulic diameter of outer annular area

geom = ConcentricGeometry(N_tube=4, L_tube=0.2, d_i=d_i, d_m=d_m, d_o=d_o, passes=5)

AS_i = np.pi * d_i * geom.L_tube * geom.passes # Surface area of inner tube
AS_o = np.pi * d_o * geom.L_tube * geom.passes # Surface area of outer annulus

V_i = mdot_i / (rho * AX_i * geom.N_tube)
Re_i = rho * V_i * d_i / mu
Nu_i = 0.023 * Re_i**0.8 * Pr**0.3

V_o = mdot_o / (rho * AX_o * geom.N_tube)
Re_o = rho * V_o * d_oh / mu
Nu_o = 0.023 * Re_o**0.8 * Pr**0.3

h_i = Nu_i * k_w / d_i # 8400
h_o = Nu_o * k_w / d_m # 11000

H = 1 / ((1/h_i)  + ((1/h_o)*(d_i/d_m))) # Neglect resistance of conduction through copper
A = geom.N_tube * geom.passes * geom.L_tube * np.pi * d_i

C_min = cp * min(mdot_i, mdot_o)
C_max = cp * max(mdot_i, mdot_o)
R_c = C_min / C_max

NTU = H * A / C_min
effectiveness = (1 - np.exp(-NTU * (1 - R_c))) / (1 - R_c * np.exp(-NTU * (1 - R_c)))
Q_max = C_min * (T_tube_in - T_shell_in)
Q = effectiveness * Q_max 

print(Q)