"""Shin, Lee, Chung, Lee 2025: SASSHA, sharpness-aware adaptive second-order optimization with stable Hessian approximation.

Algorithm 1: g = grad f_B(x); eps = rho g/|g|; g~ = grad f_B(x + eps); m = EMA(g~); every k steps D = EMA(|diag H(x + eps)|)
(absolute values, Eq. 5); update x <- x - eta m_hat / sqrt(D_hat) (square-root preconditioner, Sec. 4.2; lazy Hessian, Sec. 4.3).
Toy (deterministic, beta1 = beta2 = 0, k = 1, exact diagonal Hessian): the equal-depth two-well loss of the SAM page.
Diagonal-Newton step x <- x - eta g/|diag H| (AdaHessian / Sophia-H style, without their EMAs and clipping) vs SASSHA.
1-D linearization at a minimum with curvature a, ignoring the SAM term: Newton multiplies by 1 - eta (any a is stable for
eta < 2); the square-root step multiplies by 1 - eta sqrt(a), so minima with a > 4/eta^2 are unstable.
"""
from common import *

print("shin2025sassha")
CS, SS, CF, SF, B, AF = np.array([-.55, .35]), .16, np.array([.9, -.45]), .55, .04, 1.035


def L(z):
    z = np.asarray(z, float)
    return 1 + B * z @ z - np.exp(-((z - CS) ** 2).sum() / (2 * SS ** 2)) - AF * np.exp(-((z - CF) ** 2).sum() / (2 * SF ** 2))


def G(z):
    z = np.asarray(z, float)
    es = np.exp(-((z - CS) ** 2).sum() / (2 * SS ** 2)); ef = AF * np.exp(-((z - CF) ** 2).sum() / (2 * SF ** 2))
    return 2 * B * z + es * (z - CS) / SS ** 2 + ef * (z - CF) / SF ** 2


def Hdiag(z):
    z = np.asarray(z, float)
    es = np.exp(-((z - CS) ** 2).sum() / (2 * SS ** 2)); ef = AF * np.exp(-((z - CF) ** 2).sum() / (2 * SF ** 2))
    return 2 * B + es * (1 / SS ** 2 - (z - CS) ** 2 / SS ** 4) + ef * (1 / SF ** 2 - (z - CF) ** 2 / SF ** 4)


check_grad("two-well loss", L, G, rand_pts(6, -1.5, 1.5))
for p in rand_pts(4, -1.2, 1.2, seed=3):
    check(f"diagonal Hessian at {np.round(p, 2)}", np.allclose(np.diag(fd_hess(L, p)), Hdiag(p), atol=1e-4))
ms, mf = CS.copy(), CF.copy()
for _ in range(6000): ms = ms - .01 * G(ms); mf = mf - .01 * G(mf)
a_s, a_f = Hdiag(ms).max(), Hdiag(mf).max()
check("sharp well diag curvature >> flat", a_s > 8 * a_f, f"max diag H: sharp {a_s:.2f}, flat {a_f:.2f}")
DELTA = 1e-3


def newton(z, eta):
    return z - eta * G(z) / (np.abs(Hdiag(z)) + DELTA)


def sassha(z, eta, rho):
    g = G(z); e = rho * g / (np.linalg.norm(g) + 1e-12)
    return z - eta * G(z + e) / np.sqrt(np.abs(Hdiag(z + e)) + DELTA)


ETA, RHO, START = .5, .05, np.array([-.3, .2])
zn, zs = START.copy(), START.copy()
for _ in range(3000): zn = newton(zn, ETA); zs = sassha(zs, ETA, RHO)
check("from (-0.3, 0.2): the diagonal-Newton step settles in the sharp well", np.linalg.norm(zn - ms) < 1e-3, f"end {np.round(zn, 3)}")
check("from (-0.3, 0.2): SASSHA (eta 0.5, rho 0.05) ends in the flat well", np.linalg.norm(zs - mf) < .1, f"end {np.round(zs, 3)}")
check("1-D linearization: eta sqrt(a) > 2 in the sharp well, < 2 in the flat one", ETA * np.sqrt(a_s) > 2 > ETA * np.sqrt(a_f), f"eta sqrt(a): {ETA * np.sqrt(a_s):.2f} vs {ETA * np.sqrt(a_f):.2f}")
zs0 = START.copy()
for _ in range(3000): zs0 = sassha(zs0, ETA, 0)
check("square-root preconditioning alone (rho = 0) also leaves the sharp well here", np.linalg.norm(zs0 - mf) < .1, f"end {np.round(zs0, 3)}")


def end(eta, rho, T=6000):
    z = START.copy()
    for _ in range(T): z = sassha(z, eta, rho)
    return "sharp" if np.linalg.norm(z - ms) < .1 else "flat" if np.linalg.norm(z - mf) < .1 else "other"


check("eta = 0.1: SASSHA stays in the sharp well for rho in {0, 0.05}", end(.1, 0) == "sharp" and end(.1, .05) == "sharp")
check("eta = 0.2: square root alone stays sharp, adding the SAM perturbation (rho = 0.05) reaches the flat well", end(.2, 0) == "sharp" and end(.2, .05) == "flat")
# notes example
g = np.array([1., 1.]); Hd = np.array([.01, 4.])
check("notes: H^-1 g = (100, 0.25), |H|^-1/2 g = (10, 0.5)", np.allclose(g / Hd, [100, .25]) and np.allclose(g / np.sqrt(Hd), [10, .5]))
save_json("shin2025sassha", "data.json", {"sharp": r(ms), "flat": r(mf), "aS": round(float(a_s), 3), "aF": round(float(a_f), 3),
                                          "table1": {"opt": ["SGD", "AdaHessian", "Sophia-H", "Shampoo", "SASSHA"], "lmax": [265, 11992, 22797, 436374, 107], "acc": [69.32, 68.06, 67.76, 64.08, 72.14]}})
done()
