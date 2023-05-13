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
        0.0006720430107527015,0.9996760758907913,0,0.9986580286904211,0.0006720430107527153,0.9996760758907913,0.0006720430107527015,0.9986580286904209,
        0.20295698924731184,0.6433595557612214,0.2022849462365591,0.6321610365571494,0.2022849462365592,0.6199444701527068,0.2029569892473119,0.6148542341508559,
        0.40255376344086025,0.36136048125867637,0.4018817204301074,0.3216566404442387,0.4018817204301075,0.3196205460434982,0.4018817204301075,0.30638593243868595,
        0.6008064516129031,0.16487737158722782,0.6008064516129031,0.09666820916242513,0.6008064516129031,0.09463211476168465,0.6014784946236558,0.0834335955576122,
        0.7997311827956992,0.04067561314206358,0.8004032258064517,-0.0489125404905133,0.8010752688172038,-0.04993058769088399,0.8004032258064517,-0.06214715409532623,
        1.0000000000000002,-0.0010643220731145764,1.0006720430107525,-0.11508560851457661,1.0000000000000002,-0.11508560851457683,0.9999999999999998,-0.12933826931975956
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

    K_e_turb_x = np.zeros((24,2))
    K_c_turb_x = np.zeros((24,2))
    K_e_turb_y = np.zeros(24)
    K_c_turb_y = np.zeros(24)

    Re_list = [1e10, 10000, 5000, 3000]
    for i in range(24):
        # Init sigmas
        K_e_turb_x[i,0] = K_e_turb_dataset[i * 2]
        K_c_turb_x[i,0] = K_c_turb_dataset[i * 2]
        # Init Reynolds numbers
        Re = Re_list[i % 4]
        K_e_turb_x[i,1] = Re
        K_c_turb_x[i,1] = Re
        # Init Ks
        K_e_turb_y[i] = K_e_turb_dataset[i * 2 + 1]
        K_c_turb_y[i] = K_c_turb_dataset[i * 2 + 1]

    popt, pcov = curve_fit(approximant_curve_fit, K_e_turb_x, K_e_turb_y)
    q = np.reshape(popt, (4,3))
    print(q)

    K_e_x_coords = np.reshape(K_e_turb_x, (6,4,2))[:,:,0]
    K_e_y_coords = np.reshape(K_e_turb_y, (6,4))
    plt.plot(K_e_x_coords, K_e_y_coords, 'o')
    K_e_x_coords = np.linspace(0,1,100)
    K_e_y_coords = np.zeros((100,4))
    for i in range(4):
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
