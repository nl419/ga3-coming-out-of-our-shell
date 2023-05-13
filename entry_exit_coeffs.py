import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def approximant(sigma, Re, q):
    # https://stackoverflow.com/questions/29815094/rational-function-curve-fitting-in-python
    # Approximate function as `output = (p0 sigma^2 + p1 sigma + p2) / (p3 sigma + 1)`
    # Approximate coefficients as `pi = (qi0 Re + qi1) / (qi2 Re + 1)`
    # Note: not `(p0 sigma^2 + p1 sigma + p2) / (p3 sigma + p4)`, because `p4` is redundant.
    if type(sigma) is np.ndarray:
        if type(Re) is np.ndarray:
            assert np.shape(Re) == np.shape(sigma)
            Re_array = Re
        else:
            Re_array = np.ones_like(sigma) * Re
        point_count = np.shape(sigma)[0]
        p = np.zeros((point_count, 4))
        for i,qi in enumerate(q):
            a = np.polyval(qi[:2], Re_array)
            b = np.polyval(qi[2:], Re_array) * Re_array + 1
            p[:,i] = a / b
        out = np.zeros(point_count)
        for i,s in enumerate(sigma):
            out[i] = np.polyval(p[i,:3], s) / (np.polyval(p[i,3:], s) * s + 1)
    else:
        assert type(Re) is not np.ndarray
        p = np.zeros(4)
        for i,qi in enumerate(q):
            a = np.polyval(qi[:2], Re)
            b = 1 + np.polyval(qi[2:], Re) * Re
            p[i] = a / b
        out = np.polyval(p[:3], sigma) / (1 + np.polyval(p[3:], sigma) * sigma)
    return out

def calibrate():
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
        0.0013440860215053752,0.400046274872744,0.0006720430107527015,0.5028690421101341,0.0006720430107527015,0.5222119389171678,0.0006720430107526737,0.5364645997223505,
        0.2029569892473118,0.3267468764460897,0.2022849462365591,0.421425266080518,0.2022849462365591,0.4417862100879222,0.2029569892473118,0.4540027764923644,
        0.4018817204301075,0.2442850532161034,0.4018817204301075,0.33998149005090217,0.401209677419355,0.3593243868579359,0.4018817204301076,0.37154095326237835,
        0.6008064516129031,0.162841277186488,0.6008064516129032,0.2585377140212861,0.6008064516129031,0.27584451642757957,0.6008064516129031,0.292133271633503,
        0.8004032258064517,0.08037945395650214,0.8010752688172038,0.17607589079129982,0.800403225806452,0.19338269319759283,0.799731182795699,0.20967144840351692,
        1.0000000000000002,-0.0020823692734843746,1.0000000000000002,0.09463211476168376,1.0000000000000002,0.11092086996760742,1.0000000000000002,0.12720962517353018
    ]

    def approximant_curve_fit(sigma_re, q00, q01, q02, q10, q11, q12, q20, q21, q22, q30, q31, q32):
        q = np.array([
            [q00, q01, q02],
            [q10, q11, q12],
            [q20, q21, q22],
            [q30, q31, q32],
        ])
        return approximant(sigma_re[:,0], sigma_re[:,1], q)

    n_datapoints = 11
    n_re = 4
    n_total = n_datapoints * n_re

    K_e_turb_x = np.zeros((n_total,2))
    K_c_turb_x = np.zeros((n_total,2))
    K_e_turb_y = np.zeros(n_total)
    K_c_turb_y = np.zeros(n_total)

    Re_list = [1e10, 10000, 5000, 3000]
    for i in range(n_total):
        # Init sigmas
        K_e_turb_x[i,0] = K_e_turb_dataset[i * 2]
        # K_c_turb_x[i,0] = K_c_turb_dataset[i * 2]
        # Init Reynolds numbers
        Re = Re_list[i % n_re]
        K_e_turb_x[i,1] = Re
        # K_c_turb_x[i,1] = Re
        # Init Ks
        K_e_turb_y[i] = K_e_turb_dataset[i * 2 + 1]
        # K_c_turb_y[i] = K_c_turb_dataset[i * 2 + 1]

    popt, pcov = curve_fit(approximant_curve_fit, K_e_turb_x, K_e_turb_y, maxfev = 20000)
    q = np.reshape(popt, (4,3))
    print(q)

    K_e_x_coords = np.reshape(K_e_turb_x, (n_datapoints,n_re,2))[:,:,0]
    K_e_y_coords = np.reshape(K_e_turb_y, (n_datapoints,n_re))
    plt.plot(K_e_x_coords, K_e_y_coords, 'o')
    K_e_x_coords = np.linspace(0,1,100)
    K_e_y_coords = np.zeros((100,n_re))
    for i in range(n_re):
        K_e_y_coords[:,i] = approximant(K_e_x_coords, Re_list[i], q)
    plt.plot(K_e_x_coords, K_e_y_coords)
    plt.show()

def simple_usage():
    q = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])

    print(approximant(1, 1, q))

    sigma = np.array([1,2,3,4,5,6])
    re = np.array([1,2,3,4,5,6])

    print(approximant(sigma, re, q))

calibrate()
