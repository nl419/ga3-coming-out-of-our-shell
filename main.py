import numpy as np
from scipy import interpolate

# Config
L_tube = 0.35
d_i = 0.006
d_o = 0.008
d_noz = 0.02

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
A_tube = (np.pi/4)*(d_i)**2
m_tube = V_tube*rho*A_tube
V_noz_2 = 2
A_noz_2 = (np.pi/4)*(d_noz)**2
m_dot_2 = rho*A_noz_2*V_noz_2
Re_tube = rho*V_tube*d_i/mu
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
m_dot_1 = V_sh*rho*A_sh
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
Q_dot_LMTD = H*A*del_T_lm*F

#now using the effectiveness NTU method 
#finding C_min
if m_dot_1 < m_dot_2:
    
    C_min = cp*m_dot_1
    C_max = cp*m_dot_2
    
else:
    
    C_min = cp*m_dot_2
    C_max = cp*m_dot_1
    
    
Q_max = C_min*(T2_in - T1_in)
NTU = H*A/C_min
R_c = C_min/C_max
effectiveness = (1 - np.exp((-NTU)*(1 - R_c)))/(1 - R_c*np.exp((-NTU)*(1 - R_c)))
Q_dot_ENTU = effectiveness*Q_max


#linear interpolation for the compressor characteristic
flow_rate_cold = [0.7083, 0.6417, 0.5750, 0.5080, 0.4250, 0.3583, 0.3083, 0.2417, 0.1917, 0.1583]
comp_p_rise_cold = [0.1310, 0.2017, 0.2750, 0.3417, 0.4038, 0.4503, 0.4856, 0.5352, 0.5717, 0.5876]

flow_rate_hot = [0.4722, 0.4340, 0.3924, 0.3507, 0.3021, 0.2535, 0.1979, 0.1493, 0.1111, 0.0694]
comp_p_rise_hot = [0.0538, 0.1192, 0.1727, 0.2270, 0.2814, 0.3366, 0.3907, 0.4456, 0.4791, 0.5115]

def comp_chic(f1, f2): #will give the correpsonding pressure rises for the hot and cold side given the mass flow rates of the hot and cold streams 
    
    i = 0
    
    while f1 <= flow_rate_cold[i]:
    
        i = i + 1
        
    
    p1 = comp_p_rise_cold[i] + ((flow_rate_cold[i] - f1)/(flow_rate_cold[i] - flow_rate_cold[i-1]))*(comp_p_rise_cold[i-1]-comp_p_rise_cold[i])
    
    m = 0
    
    while f2 <= flow_rate_hot[m]:
    
        m = m + 1
        
    
    p2 = comp_p_rise_hot[m] + ((flow_rate_hot[m] - f2)/(flow_rate_hot[m] - flow_rate_hot[m-1]))*(comp_p_rise_hot[m-1]-comp_p_rise_hot[m])
    
    return p1, p2


#another linear interpolation method
def comp(f1,f2):

    flow_rate_cold = [0.7083, 0.6417, 0.5750, 0.5080, 0.4250, 0.3583, 0.3083, 0.2417, 0.1917, 0.1583]
    comp_p_rise_cold = [0.1310, 0.2017, 0.2750, 0.3417, 0.4038, 0.4503, 0.4856, 0.5352, 0.5717, 0.5876]
    
    flow_rate_hot = [0.4722, 0.4340, 0.3924, 0.3507, 0.3021, 0.2535, 0.1979, 0.1493, 0.1111, 0.0694]
    comp_p_rise_hot = [0.0538, 0.1192, 0.1727, 0.2270, 0.2814, 0.3366, 0.3907, 0.4456, 0.4791, 0.5115]

    f = interpolate.interp1d(flow_rate_cold, comp_p_rise_cold)
    g = interpolate.interp1d(flow_rate_hot, comp_p_rise_hot)
    
    comp_p_rise_cold_new = f(f1)
    comp_p_rise_hot_new = g(f2)

    return comp_p_rise_cold_new, comp_p_rise_hot_new


