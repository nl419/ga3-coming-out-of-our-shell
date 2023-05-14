import numpy as np
import matplotlib.pyplot as plt

Re_inv_classes = np.reciprocal(np.array([1e30, 10000, 5000, 3000]))
R_classes = np.array([0.2, 0.4, 0.6, 0.8, 1.1, 1.5, 2.0, 3.0, 5.0])

polys_K_e = np.array([])
polys_K_c = np.array([])
F_2_xs = np.zeros((len(R_classes), 20))
F_2_ys = np.zeros_like(F_2_xs)
F_1_xs = np.zeros((len(R_classes), 22))
F_1_ys = np.zeros_like(F_1_xs)

# Create the xs and ys for each F, then lerp.
# Need to create the xs and ys separately for each class

def _split_dataset(dataset: list, n_classes: int):
    assert(len(dataset) % n_classes == 0)
    n_datapoints = int((len(dataset) / 2) / n_classes)
    assert(n_datapoints * n_classes * 2 == len(dataset))

    dataset_x = np.zeros((n_datapoints, n_classes))
    dataset_y = np.zeros((n_datapoints, n_classes))

    for i in range(n_datapoints):
        for j in range(n_classes):
            # Init x
            dataset_x[i,j] = dataset[(i * n_classes + j) * 2]
            # Init Ks
            dataset_y[i,j] = dataset[(i * n_classes + j) * 2 + 1]
    return dataset_x, dataset_y

def _find_polys(dataset: list, n_classes: int, poly_order: int, do_plots = False):
    # Function which takes in a list of datapoints (from the digitiser)
    # Outputs the polynomials for each class
    # Later, use a lerping function between the polynomials
    # Use 1/Re to lerp the K_c and K_e
    # Simply use R to lerp the correction factor

    # First step: polynomial given a list of classes

    dataset_x, dataset_y = _split_dataset(dataset, n_classes)
    
    # Find polynomial coefficients for each class
    polys = []
    for i in range(n_classes):
        polys.append(np.polynomial.Polynomial.fit(dataset_x[:,i], dataset_y[:,i], poly_order))
        
        if do_plots:
            xs = np.linspace(np.min(dataset_x),np.max(dataset_x),100)
            ys = polys[i](xs)
            
            plt.plot(dataset_x[:,i], dataset_y[:,i], "o")
            plt.plot(xs,ys)
    if do_plots:
        plt.show()
    return np.array(polys)

def _init_all_interpolants(do_plots):
    # find the polys for all the things
    # F one_pass
    # F two_pass
    # K_c
    # K_e

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

    one_pass_dataset = [
        0.37456433454994087,0.9921968663594469,0.39440703233434427,0.9795724555628702,0.38250141366370216,0.9686353653719552,0.36563512054695924,0.9587960500329163,0.32198118542127174,0.9527481237656352,0.2763429805171438,0.9529615799868334,0.2321929779468462,0.9546495782312925,0.17117668225980567,0.962611156462585,0.1205778029095769,0.9644133190118152,
        0.40532051611576614,0.9903265832784726,0.42516321390016953,0.974530822909809,0.41325759522952754,0.9601987623436471,0.3963913021127846,0.9469644766293613,0.352737366987097,0.9364542988808426,0.30461882485991876,0.9351649607535322,0.2579884850665707,0.9350063079777365,0.19349971726725954,0.9424096145124716,0.13545982624787947,0.9436572912801484,
        0.4360766976815914,0.9879074127715601,0.4559193954659949,0.968482896642528,0.4440137767953528,0.9499122053982882,0.4271474836786099,0.9317074391046741,0.3805171438852618,0.9174213224489796,0.3279339947565928,0.9155559183673468,0.27683904796175396,0.9154574489795918,0.20887780805017217,0.9222215384615384,0.1448851076954711,0.9230395918367347,
        0.4668328792474168,0.9852747860434495,0.4866755770318202,0.9612152205398288,0.4747699583611782,0.9374097695852535,0.45691153035521503,0.9135892470091485,0.40284017889271573,0.898052,0.34678455765177596,0.8953152460984393,0.29122500385544636,0.8963104552590266,0.22028735927620416,0.9019435102040816,0.1570676759368735,0.877247387755102,
        0.49758906081324206,0.9820424489795918,0.5174317585976456,0.952260223831468,0.5055261399270035,0.9215631863067807,0.4827070374749395,0.8937316770186334,0.42119467434328883,0.8787816566626651,0.3621626484346887,0.8750202915451895,0.30313062252608847,0.8769415955473098,0.22921657327918568,0.8816036734693877,0.15083791703079216,0.9048424489795918,
        0.5283452423790674,0.9781900724160631,0.5481879401634708,0.9413434628044766,0.5342980517143884,0.9031035525321239,0.5035418701485631,0.8739126960257787,0.4365727651262016,0.8591361229829451,0.3826432940934561,0.8404250884353741,0.31835985023732427,0.844822074829932,0.24130404050789087,0.843781581632653,0.16175140081221404,0.8507499319727891,
        0.5591014239448927,0.9735753522053983,0.5789441217292961,0.9279058854509545,0.5586053565002826,0.8839454545454545,0.5291168457307357,0.8434905231734251,0.4539036652444354,0.8349528163265305,0.3700997275484501,0.8629338775510204,0.3090834318614095,0.8657697959183673,0.2482901472931743,0.815353469387755,0.16472780547987456,0.8302157823129251,
        0.589857605510718,0.9680458196181698,0.6097003032951215,0.9106366303189897,0.5789441217292961,0.8645757250268528,0.5454421858893366,0.8194166272824919,0.4663368118028067,0.8132737959183673,0.3944923194833234,0.8146001731601731,0.3284052159906785,0.8185044897959184,0.23417724772528656,0.8681330612244897,0.16704278688805493,0.8065306122448979,
        0.6513699686423686,0.9531140487162606,0.6379761476378963,0.8919232653061224,0.5963064822906491,0.8447794047619047,0.5591014239448927,0.7972207792207792,0.4757620932503983,0.7941015873015873,0.40383231378193585,0.7923510204081632,0.33636714131496415,0.7939790476190476,0.25302781062046975,0.7887273469387754,0.1695644630648229,0.7825434693877551,
        0.7128823317740193,0.929989624753127,0.6612913175345704,0.8723719339164236,0.6111885056289517,0.8256843148688047,0.5695188402817045,0.7769425306122448,0.4841952398087698,0.7747053061224489,0.4112733254510872,0.7707890379008746,0.34182388320567514,0.77376,0.256925483399549,0.7670528279883382,0.17117668225980567,0.7578788571428572,
        0.7709222227933993,0.8939057823129251,0.6886768663486907,0.8437671155099726,0.636487945304066,0.7847599257884972,0.5789441217292961,0.7562893424036281,0.4921323189225312,0.7543418367346939,0.4177222022310183,0.7497597278911564,0.3464125070683184,0.7514271428571428,0.25997275484501103,0.7454533333333333,0.1724522842602316,0.7354075801749271,
        0.8091194160283759,0.854602990913848,0.7138744666632395,0.8086636106750392,0.6563306430884696,0.7431250793650793,0.5873772682876676,0.735278163265306,0.4995733305916824,0.732751720116618,0.4277144179010214,0.7085788921282798,0.35064285999874334,0.7313262585034013,0.26245309206806144,0.7246671020408163,0.17378103634400863,0.7079036734693878,
        0.8276268441885569,0.8281544897959183,0.7361975016706933,0.7679936326530612,0.6712126664267721,0.7029070553935859,0.60126715673675,0.6937765986394557,0.5114789492623245,0.6912662857142857,0.4359664604716781,0.6677456689342404,0.35429643609872874,0.7108746355685132,0.26476807347624176,0.6998860770975056,0.17536569623651307,0.6767640060468632,
        0.8418598673726416,0.8006136961451247,0.7530637947874362,0.7279914285714285,0.6836143525420242,0.6613736054421768,0.612676707962782,0.6520045714285714,0.520408163265306,0.6493262040816326,0.4396200365716635,0.6460086297376093,0.358194108877808,0.6891551020408163,0.26778581709761984,0.67147,0.17650940728936407,0.6430714285714285,
        0.8577340256001643,0.7605257142857142,0.7669536832365186,0.6844285714285714,0.6930396339896159,0.6180050612244898,0.6214956847558502,0.6086815419501134,0.5289121766014789,0.6054504956268221,0.4434468425729413,0.6246267055393586,0.36209178165688727,0.6643295626822157,0.2693980362926026,0.6465217959183673,0.17679877996538665,0.6159055479969766,
        0.8696396442708064,0.7191843265306122,0.7772293660177276,0.6413721282798834,0.7015436473257888,0.5742103790087463,0.6290469336349148,0.5671318367346938,0.5367075221596375,0.557780058309038,0.4494705186860638,0.5821254421768707,0.36733592321419384,0.6190448979591836,0.2722090851453931,0.5978280272108843,0.17849548795932366,0.5910187198515771,
        0.8793322366730064,0.6758814285714285,0.7853081786870918,0.6001387755102041,0.7084177247725285,0.5306062585034013,0.6348343871553659,0.5249344217687075,0.5392587261604893,0.5345975510204082,0.4517704677474379,0.5576,0.3725800647715005,0.5567177142857143,0.2738626432940934,0.5683134693877551,0.17779091485460682,0.5613287074829931,
        0.8859390031651376,0.631198833819242,0.7984264066667807,0.5109298866213152,0.34319899244332486,0.9775999999999999,0.32808564231738024,0.9703999999999999,0.2928211586901762,0.9647999999999999,0.4549272605767747,0.5315365597667638,0.375556469439161,0.521993469387755,0.2751303712080969,0.5314815419501133,0.17894840555869698,0.5372234013605441,
        0.34068010075566746,0.9939999999999999,0.35831234256926947,0.9847999999999999,0.31045340050377823,0.9823999999999999,0.28652392947103267,0.98,0.24496221662468506,0.9792,0.24748110831234252,0.9672,0.20151133501259447,0.9707999999999999,0.14357682619647363,0.9783999999999999,0.10012594458438287,0.9815999999999999,
        0.28400503778337527,0.9964,0.2915617128463476,0.9912,0.2651133501259445,0.9887999999999999,0.24999999999999992,0.9863999999999999,0.19836272040302264,0.988,0.2096977329974811,0.9795999999999999,0.13916876574307305,0.9899999999999999,0.1070528967254408,0.9907999999999999,0.0698992443324937,0.9932,
        0.2311083123425692,0.9979999999999999,0.23488664987405541,0.9952,0.22481108312342565,0.9927999999999999,0.21599496221662465,0.9904,0.1630982367758186,0.9927999999999999,0.1542821158690176,0.9907999999999999,0.11460957178841305,0.9935999999999999,0.07745591939546598,0.9959999999999999,0.0428211586901763,0.9979999999999999,
        2.7755575615628914e-17,0.9999999999999999,0,0.9999999999999999,0,0.9999999999999999,-2.7755575615628914e-17,0.9999999999999999,0,1.0004,2.7755575615628914e-17,0.9999999999999999,0,0.9999999999999999,2.7755575615628914e-17,0.9999999999999999,0,0.9999999999999999
    ]
    
    two_pass_dataset = [
        0.6333573305680111,0.9905479864946897,0.6293799395788878,0.9765391565790162,0.5882802326912797,0.9670280069602748,0.519338788879808,0.9673003435654102,0.455037634555647,0.9640941989867715,0.3721753222822436,0.9682824058687769,0.315166051438142,0.967056891145668,0.23031504367017688,0.9727470756681168,0.1580591073677691,0.9729541415166494,
        0.6671651539755598,0.9880185571166903,0.6631877629864364,0.9711006771007077,0.6220880560988283,0.9586227090108708,0.5531466122873566,0.957859341254052,0.4888454579631956,0.9518514306922787,0.4059831456897922,0.9544840178752536,0.34731662860022255,0.9492996654069412,0.25782533134494684,0.9531142111724351,0.17396867132426255,0.9510426735012708,
        0.7009729773831084,0.9852704331921429,0.696995586393985,0.9639250201866116,0.6558958795063768,0.9474899186979144,0.5869544356949052,0.9455010362179866,0.5226532813707442,0.9349459296128931,0.4378022736027791,0.9355635010053465,0.3725067715313372,0.9273469130671776,0.27472924304872115,0.9316044854289919,0.1815920040534157,0.9290514926365929,
        0.734780800790657,0.981808457257165,0.7308034098015336,0.9546738282364085,0.6897037029139255,0.932858015640189,0.6207622591024539,0.9290700610414885,0.5528151630382631,0.9142837307497147,0.4633238657829874,0.9138996742417909,0.3894106832351115,0.905787639195103,0.2853356190197168,0.9103950068459846,0.18590084429163264,0.9054235412663036,
        0.8023964476057541,0.9714885504474157,0.7984190566166308,0.92569473766269,0.7225171785741933,0.9139215952753482,0.6519184885172535,0.9089151370857489,0.5763480597239096,0.892824934005831,0.4818850237322297,0.892434598182488,0.40167430545157523,0.8843851946602495,0.29295895174886993,0.8884151574694226,0.18884705983913144,0.8844261139228897,
        0.8700120944208514,0.9515543362139492,0.851450936471609,0.883164268680879,0.7725660151873289,0.8704579305538205,0.6771086314483682,0.8875848684311215,0.594909217673152,0.8711757552068923,0.4961373414432551,0.871185501990655,0.41128633367529,0.8626246905179283,0.29859358898346133,0.866084263216131,0.19103830765258367,0.8587901958787524,
        0.9260270175176721,0.9130353592043889,0.8816128181391278,0.8408295623841424,0.8037222446021286,0.8275132651992045,0.696995586393985,0.8661722257126911,0.6098244338823646,0.8496517554093874,0.5077380651615315,0.8496885827457636,0.41890966640444316,0.8410657249856663,0.3030129123047095,0.8414959618042841,0.19244696696123154,0.8351154796368743,
        0.9525429574451612,0.86960164962921,0.9014997730847447,0.7971745547559088,0.8252664457932135,0.7845229868171275,0.7268260188124103,0.8231825919591698,0.6220880560988283,0.8283426539512075,0.5173500933852464,0.8279827125640007,0.4252072021372218,0.8188851463910534,0.305590850908771,0.8212935372778815,0.19385562626987937,0.8061323659087617,
        0.9661323766579994,0.8249879621334042,0.9150891922975828,0.753086972065478,0.8415074589988005,0.7399120323332975,0.7483702200034951,0.7794874122513946,0.6419750110444451,0.7840003508346449,0.5253048753634931,0.8064116797327063,0.43051039012271963,0.7963979238527361,0.3088316880110196,0.7980747785339951,0.19461322455352195,0.7775483402544409,
        0.9733639966382237,0.7811203869070393,0.9250326697703912,0.7066932966448198,0.8537710812152642,0.6945139704003871,0.7646112332090822,0.7349188194531924,0.6568902272536578,0.7391487020884079,0.532265309594459,0.7841832734285138,0.4350678172977568,0.7723511763556286,0.31118866044901866,0.7730689336258187,0.19554970020969117,0.7492054877235999,
        0.9787274481235566,0.7357605871417925,0.9319931040013572,0.6609063954330276,0.8631621432729167,0.6470008608660697,0.7768748554255459,0.691094289165449,0.6681595017228406,0.6945665808809245,0.5432031348145482,0.7405406250877717,0.43786253710261436,0.7517350421413628,0.31273542361145557,0.7495094593889275,0.19559573482762085,0.7218188097993531,
        0.9819590783022194,0.6863856511573422,0.9376277412359486,0.6137042722947763,0.8696806451717577,0.6044237835279602,0.7864868836492607,0.6457440549421181,0.6770066470640317,0.6451248700557921,0.5505552817944429,0.6966949139572145,0.44443125858465143,0.7061934970797136,0.3144202906276814,0.7241073657027346,0.1961757710135347,0.6958292324138247,
        0.9846935346072416,0.6300586159590071,0.9416051322250719,0.5650454234065312,0.8759781809045364,0.5547016183719624,0.793718503629485,0.6009821015466157,0.6847319641775212,0.595501046028799,0.5574922802199174,0.6517241128965481,0.4491452034606494,0.6554185377520707,0.31582894993632926,0.6961975057775873,0.1962586333258081,0.6678404567678711,
        0.9861666423809912,0.5734864827283266,0.9453836536647391,0.5100616256358877,0.8802870211427534,0.5080624273754625,0.8000394751677551,0.5559262734466965,0.6903666014121127,0.5488618550322991,0.5630900897601651,0.6056285635322691,0.4530489390610854,0.6042343857980251,0.31715474693270373,0.6695615710597409,0.19683866951172188,0.6374000327288695,
        0.9872788387501719,0.5126711054705482,0.5811390079608082,0.9828926905132191,0.543172075933864,0.9762830482115085,0.8053794908475966,0.5091567253706426,0.6944765721008735,0.5062526268449723,0.5672884469153509,0.5530969987155685,0.455866257678381,0.5487303288309553,0.3177439900422035,0.6447327645940082,0.19683866951172188,0.6160191334384266,
        0.5811390079608082,0.9926127527216173,0.5076546233925291,0.9895023328149298,0.5088793631353337,0.9813374805598755,0.4623392529087568,0.9790046656298599,0.39804041641151255,0.9782270606531881,0.5707134224893181,0.5084482375660709,0.4570263300502087,0.5069260409958524,0.31866994349998884,0.6224827067616061,0.14145744029393745,0.9844479004665628,
        0.5284751990202081,0.9949455676516328,0.41702388242498467,0.9945567651632969,0.4709124311083895,0.9856143079315707,0.38273116962645437,0.988724727838258,0.3508879363135334,0.9860031104199065,0.33619105939987753,0.9782270606531881,0.27250459277403555,0.9817262830482114,0.19534598897734232,0.9860031104199065,0.12063686466625839,0.9918351477449454,
        0.4292712798530312,0.9972783825816484,0.3851806491120637,0.9953343701399687,0.41457440293937536,0.9902799377916017,0.33374157991426817,0.9922239502332814,0.32578077158603796,0.988724727838258,0.2761788120024495,0.989113530326594,0.2186160440906307,0.9910575427682736,0.15370483772198407,0.993779160186625,0.09797917942437231,0.9957231726283047,
        0.37538273116962645,0.9984447900466562,1.3877787807814457e-17,0.9999999999999999,0.3717085119412125,0.9930015552099533,0.2896509491733007,0.9949455676516329,0.28842620943049607,0.9922239502332814,0.23821187997550522,0.9930015552099533,0.18554807103490506,0.9945567651632969,0.11941212492345377,0.9968895800933124,1.3877787807814457e-17,0.9999999999999999,
        1.3877787807814457e-17,0.9999999999999999,0.4635639926515615,0.9922239502332814,-6.938893903907228e-18,0.9999999999999999,2.0816681711721685e-17,0.9999999999999999,0.0006123698714023407,0.9999999999999999,1.3877787807814457e-17,0.9999999999999999,2.7755575615628914e-17,0.9999999999999999,2.0816681711721685e-17,0.9999999999999999,0.19718309859154934,0.5027216174183514
    ]

    global polys_K_c, polys_K_e, F_1_xs, F_1_ys, F_2_xs, F_2_ys

    polys_K_e = _find_polys(K_e_turb_dataset, len(Re_inv_classes), 4, do_plots)
    polys_K_c = _find_polys(K_c_turb_dataset, len(Re_inv_classes), 4, do_plots)
    F_1_xs, F_1_ys = _split_dataset(one_pass_dataset, len(R_classes))
    F_2_xs, F_2_ys = _split_dataset(two_pass_dataset, len(R_classes))

    if not do_plots:
        return

    # Do a sweep over many `R_inv`s
    # Do a sweep over many `R`s

    Re_invs = np.linspace(np.min(Re_inv_classes), np.max(Re_inv_classes), 10)

    # Plot all the results against the input data
    # First, the K_e and K_c
    xs = np.linspace(0, 1, 100)
    for i,Re_inv in enumerate(Re_invs):
        ys = K_c_plus_K_e(xs, 1 / Re_inv, 0) # Find K_e
        plt.plot(xs,ys,label=f"Re={1/Re_inv}")

    K_e_xs, K_e_ys = _split_dataset(K_e_turb_dataset, len(Re_inv_classes))
    K_c_xs, K_c_ys = _split_dataset(K_c_turb_dataset, len(Re_inv_classes))
    for i,_ in enumerate(Re_inv_classes):
        plt.plot(K_e_xs[:,i],K_e_ys[:,i], "o")
    plt.legend()
    plt.show()

    for i,Re_inv in enumerate(Re_invs):
        ys = K_c_plus_K_e(xs, 1 / Re_inv, 1) # Find K_c
        plt.plot(xs,ys,label=f"Re={1/Re_inv}")

    for i,_ in enumerate(Re_inv_classes):
        plt.plot(K_c_xs[:,i],K_c_ys[:,i], "o")
    plt.legend()
    plt.show()

    # Now do the correction factors also
    ##### gosh diddly darn
    Rs = np.linspace(np.min(R_classes), np.max(R_classes), 10)
    # use the same xs
    for i,_ in enumerate(R_classes):
        plt.plot(F_1_xs[:,i],F_1_ys[:,i], "o")
    for i,R in enumerate(Rs):
        ys = F_1(xs, R)
        plt.plot(xs,ys,label=f"R={R}")
    plt.legend()
    plt.show()
    for i,_ in enumerate(R_classes):
        plt.plot(F_2_xs[:,i],F_2_ys[:,i], "o")
    for i,R in enumerate(Rs):
        ys = F_2(xs, R)
        plt.plot(xs,ys,label=f"R={R}")
    plt.legend()
    plt.show()

def K_c_plus_K_e(sigma, re, _debug_switch = None):
    Re_inv = 1/re
    xs = Re_inv_classes
    # Clamp Re to the valid range of [3000,inf)
    interp_factor = np.interp(Re_inv, xs, np.arange(0, len(Re_inv_classes), 1), 0, len(Re_inv_classes) - 1)
    interp_offset = int(np.floor(interp_factor))
    interp_ratio = interp_factor - interp_offset
    if (interp_offset == len(Re_inv_classes) - 1):
        interp_offset -= 1
        interp_ratio = 1
    assert(np.shape(Re_inv_classes) == np.shape(polys_K_e))
    assert(np.shape(Re_inv_classes) == np.shape(polys_K_c))

    # Linear interpolate between the polynomials adjacent to this Reynolds number
    if _debug_switch == 0:
        return polys_K_e[interp_offset](sigma) * (1 - interp_ratio) \
             + polys_K_e[interp_offset + 1](sigma) * interp_ratio
    elif _debug_switch == 1:
        return polys_K_c[interp_offset](sigma) * (1 - interp_ratio) \
             + polys_K_c[interp_offset + 1](sigma) * interp_ratio
    else:
        return (polys_K_e[interp_offset](sigma) + polys_K_c[interp_offset](sigma)) * (1 - interp_ratio) \
             + (polys_K_e[interp_offset + 1](sigma) + polys_K_c[interp_offset + 1](sigma)) * interp_ratio

def F_1(P, R):
    xs = R_classes
    ys = np.zeros(np.shape(R_classes))
    # TODO make the error go away

    for i,_ in enumerate(ys):
        ys[i] = np.interp(P, F_1_xs[i,:], F_1_ys[i,:])
    
    return np.interp(R, xs, ys)

def F_2(P, R):
    xs = R_classes
    ys = np.zeros(np.shape(R_classes))

    for i,_ in enumerate(ys):
        ys[i] = np.interp(P, F_2_xs[i,:], F_2_ys[i,:])
    
    return np.interp(R, xs, ys)

if __name__ == "__main__":
    _init_all_interpolants(True)
else:
    _init_all_interpolants(False)