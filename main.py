import numpy as np
from scipy import interpolate

from entry_exit_coeffs import K_c_plus_K_e, F_1, F_2


# Config
L_tube = 0.35
d_i = 0.006
d_o = 0.008
d_noz = 0.02
d_sh = 0.064

# Properties
rho = 990.1
mu = 6.51e-4
Pr = 4.31
k_tube = 386
cp = 4179
k_w = 0.632
T1_in = 20
T2_in = 60
T1_out = 25 #for now, need to iterate in order to find the exact temperatures
T2_out = 50 #for now, need to iterate in order to find the exact temperatures

#Parameters that we can change in our design
m_dot_2 = 0.45
N = 13
N_b = 9
Y = 0.014


# Pressure drop for given config
## Pressure drop for tube side
m_dot_2 = 0.45
A_tube = (np.pi/4)*(d_i)**2
A_pipe = (np.pi/4)*d_sh**2
m_tube = m_dot_2/N
V_tube = m_tube/(rho*A_tube)
A_noz_2 = (np.pi/4)*(d_noz)**2
V_noz_2 = m_dot_2/(rho*A_noz_2)
Re_tube = rho*V_tube*d_i/mu
f = (1.82 * np.log10(Re_tube) - 1.64) ** (-2)
sigma = (N*A_tube)/A_pipe 



#now that sigma has been obtained, can calculate Kc and Ke
#from the entry_exit_coeffs 
Kc_Ke_sum = K_c_plus_K_e(sigma, Re_tube)
del_p_ends = 0.5*rho*(V_tube**2)*(Kc_Ke_sum)
del_p_noz_2 = rho*V_noz_2**2
del_p_tube = 0.5*rho*(V_tube**2)*(f*L_tube/d_i)
del_p_2 = del_p_ends + del_p_noz_2 + del_p_tube



## Pressure drop for shell side, this is the cold side
B = L_tube/(N_b + 1)
A_sh = (d_sh/Y)*(Y - d_o)*B
m_dot_1 = 0.5
V_sh = m_dot_1/(rho*A_sh)


A_pipe =(np.pi/4)*(d_sh)**2 
d_sh_1 = d_sh * (A_sh/A_pipe)#propose characteristic length, same as in example
a = 0.2 #this is a constant, 0.2 for triangular arrangement, 0.34 for square
Re_sh = rho*V_sh*d_sh_1/mu
del_p_sh = 4*a*Re_sh**(-0.15)*N*rho*V_sh**2
A_noz_1 = (np.pi/4)*(d_noz)**2
V_noz_1 = m_dot_1/(rho*A_noz_1)
del_p_noz_1 = rho*V_noz_1**2


del_p_1 = del_p_sh + del_p_noz_1


#now adjusting the mass flowrates according to the pressure drop terms 

#compressor characteristics
flow_rate_cold = [0.7083, 0.6417, 0.5750, 0.5080, 0.4250, 0.3583, 0.3083, 0.2417, 0.1917, 0.1583]
comp_p_rise_cold = [0.1310, 0.2017, 0.2750, 0.3417, 0.4038, 0.4503, 0.4856, 0.5352, 0.5717, 0.5876]

flow_rate_hot = [0.4722, 0.4340, 0.3924, 0.3507, 0.3021, 0.2535, 0.1979, 0.1493, 0.1111, 0.0694]
comp_p_rise_hot = [0.0538, 0.1192, 0.1727, 0.2270, 0.2814, 0.3366, 0.3907, 0.4456, 0.4791, 0.5115]

#linear interpolation function
def compressor_chic_1(p1):
     
     f = interpolate.interpid(comp_p_rise_cold, flow_rate_cold)
     comp_p_rise_cold_new = p1
     m1 = f(comp_p_rise_cold_new)
     
     return m1
 
def compressor_chic_2(p2):
     
     f = interpolate.interpid(comp_p_rise_hot, flow_rate_hot)
     comp_p_rise_hot_new = p2
     m2 = f(comp_p_rise_hot_new)
     
     return m2

#fitting a polynomial to the compressor characteristic data set for when p_drop
#is outside data set

def compressor_chic_1_poly(p1): #cold
     
     coeff = np.polyfit(comp_p_rise_cold,flow_rate_cold,5)
     poly_eq = np.poly1d(coeff)
     
     m1 = poly_eq(p1)
     
     return m1

def compressor_chic_2_poly(p2): #hot
     
     coeff = np.polyfit(comp_p_rise_hot,flow_rate_hot,5)
     poly_eq = np.poly1d(coeff)
     
     m2 = poly_eq(p2)
     
     return m2
 
 
#now finding adjusted mass flow rates
if del_p_1 > comp_p_rise_cold[9] or del_p_1 < comp_p_rise_cold[0]:
     
     m_dot_1_adjusted = compressor_chic_1_poly(del_p_1/100000)
    
else:
    
     m_dot_1_adjusted = compressor_chic_1(del_p_1)
     
if del_p_2 > comp_p_rise_hot[9] or del_p_2 < comp_p_rise_hot[0]:
     
     m_dot_2_adjusted = compressor_chic_2_poly(del_p_1/100000)
    
else:
    
     m_dot_2_adjusted = compressor_chic_2(del_p_1)
    

# Heat transfer coeff
A_i = np.pi*d_i*L_tube 
A_o = np.pi*d_o*L_tube
c = 0.2 #constant, 0.2 for tringular, 0.15 for square


#find adjusted Re due to adjusted mass flow rates
V_tube_adj = m_dot_1_adjusted/(rho*A_tube)
Re_tube_adj = rho*V_tube_adj*d_i/mu


V_sh_adj = m_dot_2_adjusted/(rho*A_sh)
Re_sh_adj = rho*V_sh_adj*L_tube/mu


Nu_i = 0.023*(Re_tube_adj**(0.8))*(Pr**(0.3))
Nu_o = c*(Re_sh_adj**0.6)*(Pr**0.3)

h_i = Nu_i*k_w/d_i
h_o = Nu_o*k_w/d_o

H = ((1/h_i) + ((A_i*np.log(d_o/d_i))/(2*np.pi*k_tube*L_tube) + ((1/h_o)*(A_i/A_o))))**(-1)

# Q
del_T_lm = ((T2_in - T1_out) - (T2_out - T1_in))/(np.log((T2_in - T1_out)/(T2_out - T1_in)))
R = (T2_in - T2_out)/(T1_out - T1_in)
P = (T1_out - T1_in)/(T2_in - T1_in)
F = F_1(P, R) #use F_1 for one pass, F_2 for 2 passes
A = N*np.pi*d_i*L_tube 
A = N*np.pi*d_i*L_tube*F
Q_dot_LMTD = H*A*del_T_lm

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


