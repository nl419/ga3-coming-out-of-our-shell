import numpy as np
import matplotlib.pyplot as plt


_n_sigma_coeffs = 4
_n_re_coeffs = 3
_sigma_re_coeffs_Ke = np.zeros((_n_sigma_coeffs, _n_re_coeffs))
_sigma_re_coeffs_Kc = np.zeros((_n_sigma_coeffs, _n_re_coeffs))

def _K_poly(sigma, Re, sigma_re_coeffs):
    assert type(Re) is not np.ndarray
    assert np.shape(sigma_re_coeffs) == (_n_sigma_coeffs, _n_re_coeffs)
    coeffs = np.zeros(_n_sigma_coeffs)
    for i,pp_coeffs_i in enumerate(sigma_re_coeffs):
        coeffs[i] = np.polynomial.Polynomial(pp_coeffs_i)(1 / Re)
    return np.polynomial.Polynomial(coeffs)(sigma)

def Kc_plus_Ke(sigma, Re):
    return _K_poly(sigma, Re, _sigma_re_coeffs_Kc) + _K_poly(sigma, Re, _sigma_re_coeffs_Ke)

def _calculate_poly_coeffs(do_plots=False):
    # https://apps.automeris.io/wpd/
    # K_e_turb_re_inf,,K_e_turb_re_10000,,K_e_turb_re_5000,,K_e_turb_re_3000,
    # X,Y,X,Y,X,Y,X,Y
    K_e_turb_dataset = [
        2.7755575615628914e-17,1.122439024390244,0.0006459948320413564,1.1230155210643016,2.7755575615628914e-17,1.1230155210643016,2.7755575615628914e-17,1.122439024390244,
        0.10465116279069775,1.0094456762749446,0.10206718346253238,1.0094456762749446,0.10335917312661505,1.0077161862527717,0.10335917312661505,1.00079822616408,
        0.20413436692506473,0.9102882483370289,0.20284237726098203,0.9056762749445677,0.20155038759689936,0.9004878048780489,0.20219638242894072,0.8947228381374724,
        0.3036175710594317,0.8215077605321509,0.3016795865633077,0.8088248337028825,0.30232558139534904,0.8047893569844794,0.3016795865633077,0.7967184035476722,
        0.401808785529716,0.7454101995565412,0.40374677002584003,0.7229268292682933,0.40180878552971605,0.7194678492239471,0.401808785529716,0.712549889135255,
        0.5025839793281657,0.6819955654101997,0.5006459948320416,0.6525942350332596,0.5019379844961244,0.6456762749445683,0.5025839793281657,0.6410643015521069,
        0.6020671834625326,0.628957871396896,0.6014211886304913,0.5932150776053219,0.6014211886304913,0.5891796008869182,0.6014211886304913,0.5811086474501105,
        0.700258397932817,0.5874501108647452,0.7015503875968997,0.5470953436807098,0.7009043927648582,0.5413303769401332,0.7009043927648584,0.533259423503326,
        0.8016795865633081,0.5563192904656322,0.8010335917312665,0.5136585365853662,0.8016795865633081,0.5032815964523284,0.8010335917312666,0.4957871396895789,
        0.8998708010335923,0.5361419068736144,0.9005167958656336,0.4871396895787147,0.8998708010335921,0.47906873614190704,0.8998708010335921,0.4698447893569846,
        1.0019379844961251,0.5315299334811535,1.0006459948320416,0.4721507760532151,1.001291989664083,0.464079822616408,1.001291989664083,0.4548558758314858
    ]

    # K_c_turb_re_inf,,K_c_turb_re_10000,,K_c_turb_re_5000,,K_c_turb_re_3000,
    # X,Y,X,Y,X,Y,X,Y
    K_c_turb_dataset = [
        2.0816681711721685e-17,0.7707760532150779,3.469446951953614e-17,0.8290022172949003,0.0006459948320413772,0.8399556541019957,3.469446951953614e-17,0.8491796008869181,
        0.10271317829457374,0.7494456762749445,0.10335917312661505,0.8053658536585366,0.10335917312661505,0.8174722838137473,0.10335917312661505,0.8243902439024391,
        0.20284237726098206,0.7252328159645238,0.20348837209302337,0.7811529933481154,0.20284237726098206,0.7932594235033262,0.20284237726098206,0.800177383592018,
        0.3016795865633077,0.7004434589800443,0.30232558139534904,0.7569401330376943,0.3016795865633077,0.7690465631929048,0.30297157622739035,0.7765410199556543,
        0.401808785529716,0.676230598669623,0.4024547803617573,0.732727272727273,0.4024547803617573,0.7448337028824833,0.401808785529716,0.7523281596452329,
        0.5019379844961243,0.6508647450110865,0.5012919896640831,0.7090909090909094,0.5019379844961244,0.7206208425720619,0.5019379844961244,0.7281152993348119,
        0.6014211886304913,0.6272283813747228,0.6014211886304913,0.6843015521064306,0.60077519379845,0.6952549889135258,0.6014211886304913,0.7044789356984479,
        0.7009043927648584,0.6035920177383597,0.7021963824289411,0.6600886917960088,0.7015503875968998,0.6698891352549889,0.7015503875968996,0.6802660753880272,
        0.8010335917312666,0.5793791574279378,0.8003875968992252,0.6364523281596456,0.8010335917312661,0.6462527716186249,0.8016795865633078,0.6560532150776057,
        0.9011627906976748,0.5551662971175165,0.9005167958656339,0.6122394678492238,0.9005167958656335,0.6214634146341467,0.9011627906976755,0.6318403547671837,
        1.0006459948320414,0.5309534368070953,1.001291989664083,0.5886031042128609,1.0012919896640833,0.5966740576496677,1.000645994832042,0.6082039911308207
    ]

    # Note: need n_datapoints = same for both K_e and K_c
    # Same goes for n_re
    n_datapoints = 11
    n_re = 4

    K_e_turb_x = np.zeros((n_datapoints, n_re))
    K_c_turb_x = np.zeros((n_datapoints, n_re))
    K_e_turb_y = np.zeros((n_datapoints, n_re))
    K_c_turb_y = np.zeros((n_datapoints, n_re))

    Re_list = np.array([1e30, 10000, 5000, 3000])
    Re_inv_list = np.reciprocal(Re_list)
    for i in range(n_datapoints):
        for j in range(n_re):
            # Init sigmas
            K_e_turb_x[i,j] = K_e_turb_dataset[(i * n_re + j) * 2]
            K_c_turb_x[i,j] = K_c_turb_dataset[(i * n_re + j) * 2]
            # Init Ks
            K_e_turb_y[i,j] = K_e_turb_dataset[(i * n_re + j) * 2 + 1]
            K_c_turb_y[i,j] = K_c_turb_dataset[(i * n_re + j) * 2 + 1]
    
    for K_turb_x, K_turb_y, sigma_re_coeffs in zip((K_e_turb_x, K_c_turb_x), (K_e_turb_y, K_c_turb_y), (_sigma_re_coeffs_Ke, _sigma_re_coeffs_Kc)):
        # Find polynomial coefficients for each Re_inv
        polys = np.zeros((n_re, _n_sigma_coeffs))
        for i,Re_inv in enumerate(Re_inv_list):
            poly = np.polynomial.Polynomial.fit(K_turb_x[:,i], K_turb_y[:,i], _n_sigma_coeffs - 1)
            polys[i,:] = poly.convert().coef
            
            if do_plots:
                xs = np.linspace(0,1,100)
                ys = poly(xs)
                
                plt.plot(K_turb_x[:,i], K_turb_y[:,i], "o")
                plt.plot(xs,ys)
        if do_plots:
            plt.show()

        # Fit a polynomial to each of the polynomial coefficients
        for i,coeffs in enumerate(polys.T):
            poly = np.polynomial.Polynomial.fit(Re_inv_list, coeffs, _n_re_coeffs - 1)
            sigma_re_coeffs[i,:] = poly.convert().coef

            # if do_plots:
            #     xs = np.linspace(Re_inv_list.min(), Re_inv_list.max(), 100)
            #     ys = poly(xs)

            #     plt.plot(Re_inv_list, coeffs, "o")
            #     plt.plot(xs, ys)
            #     plt.show()
        
        for re_inv in np.linspace(Re_inv_list.min(), Re_inv_list.max(), 10):
            xs = np.linspace(0,1,100)
            ys = _K_poly(xs, 1/re_inv, sigma_re_coeffs)

            if do_plots:
                plt.plot(xs,ys,label=f"Re={1/re_inv}")
        
        if do_plots:
            plt.legend()
            for i in range(n_re):
                plt.plot(K_turb_x[:,i], K_turb_y[:,i], "o")
            plt.show()

if __name__ == "__main__":
    _calculate_poly_coeffs(True)
else:
    _calculate_poly_coeffs()
