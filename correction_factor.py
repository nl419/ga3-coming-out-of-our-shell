import numpy as np

def correction_factor(Tin_shell, Tin_tube, Tout_shell, Tout_tube, Ns):
    R = (Tin_tube - Tout_tube) / (Tout_shell - Tin_shell)
    if R == 1:
        R -= 1e-5
    
    P = (Tout_shell - Tin_shell) / (Tin_tube - Tin_shell)
    W = ((1 - P * R) / (1 - P)) ** (1 / Ns)

    S = (R**2 + 1) ** 0.5 / (R - 1)

    F = S * np.log(W) / np.log((1 + W - S + S*W) / (1 + W + S - S*W))
    return F