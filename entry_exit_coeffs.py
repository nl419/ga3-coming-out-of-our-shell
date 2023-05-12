import numpy as np

# 4 datapoints: 3000, 5000, 10000, inf
K_e_turbulent_datapoints_px = [
    [208, 208, 208, 208], # 0.0
    [439, 433, 429, 419], # 0.3
    [624, 618, 614, 586], # 0.7
    [678, 672, 665, 625]  # 1.0
]

# 4 datapoints: inf, 10000, 5000, 3000
K_c_turbulent_datapoints_px = [
    [451, 417, 409, 404], # 0.0
    [507, 466, 457, 454], # 0.3
    [574, 535, 527, 520], # 0.7
    [626, 585, 578, 571]  # 1.0
]

#  1.5 =>    0 px
# -0.9 => 1000 px
def px_to_decimal(px):
    return (px / 1000) * (-0.9 - 1.5) + 1.5

# option 1: construct a single multidimensional polynomial (ew)
# option 2: construct many 1D polynomials, one for each Re, and construct a 1D polynomial in terms of the coefficients.
# I will do option 2.

K_e_turbulent_datapoints_decimal = np.zeros_like(K_e_turbulent_datapoints_px)

for i,y_list in enumerate(K_e_turbulent_datapoints_px):
    for j,y_px in enumerate(y_list):
        dec = px_to_decimal(y_px)
        K_e_turbulent_datapoints_decimal[i,j] = dec

print(K_e_turbulent_datapoints_decimal)

vandermonde = np.zeros((4,4))

xs = [0.0, 0.3, 0.7, 1.0]

# model as a0 + a1 x + a2 x^2 + a3 x^3
for i,x in enumerate(xs):
    for j in range(4):
        vandermonde[i,j] = x ** j

print(vandermonde)
