"""Arjevani, Carmon, Duchi, Foster, Srebro, Woodworth 2023: lower bounds for non-convex stochastic optimization.

Hard function F_T (Eq. 15) = the Carmon et al. chain; stochastic gradient (Eq. 17):
[g_T(x, z)]_i = grad_i F_T(x) (1 + 1{i > prog_{1/4}(x)} (z/p - 1)),  z ~ Bernoulli(p),  prog_a(x) = max{i : |x_i| > a} (x_0 = 1).
Lemma 3: unbiased, probability-p zero-chain, variance <= 23^2 (1 - p)/p.  Lemma 1: a zero-respecting method needs about T/p
queries to activate all T coordinates.  Theorem 1: Omega(Delta L sigma^2 / eps^4) (bounded variance); Theorem 2:
Omega(Delta Lbar sigma / eps^3 + sigma^2/eps^2) under mean-squared smoothness.
"""
from common import *
from math import erf, sqrt, e, pi

print("arjevani2023lower")
Psi = lambda t: 0.0 if t <= .5 else np.exp(1 - 1 / (2 * t - 1) ** 2)
dPsi = lambda t: 0.0 if t <= .5 else Psi(t) * 4 / (2 * t - 1) ** 3
Phi = lambda t: sqrt(e) * sqrt(2 * pi) * .5 * (1 + erf(t / sqrt(2)))
dPhi = lambda t: sqrt(e) * np.exp(-t * t / 2)


def gF(x):
    T = len(x); G = np.zeros(T)
    G[0] = -Psi(1) * dPhi(x[0])
    for i in range(1, T):
        G[i] += -Psi(-x[i - 1]) * dPhi(-x[i]) - Psi(x[i - 1]) * dPhi(x[i])
        G[i - 1] += -dPsi(-x[i - 1]) * Phi(-x[i]) - dPsi(x[i - 1]) * Phi(x[i])
    return G


def prog(x, a):
    idx = np.nonzero(np.abs(x) > a)[0]
    return int(idx[-1] + 1) if len(idx) else 0


def gs(x, z, p):
    G = gF(x).copy(); k = prog(x, .25)
    G[k:] *= (1 + (z / p - 1))      # only coordinate k+1 can be non-zero beyond prog
    return G


R = np.random.default_rng(48)
T, p = 6, .2
x = np.r_[R.normal(0, 2, 3), np.zeros(3)]
mean = np.mean([gs(x, R.random() < p, p) for _ in range(200000)], 0)
check("Eq. (17) is unbiased: E g_T(x, z) = grad F_T(x)", np.allclose(mean, gF(x), atol=2e-2 * (1 + np.abs(gF(x)).max())))
var = max(np.mean([np.sum((gs(xx, R.random() < p, p) - gF(xx)) ** 2) for _ in range(4000)]) for xx in [R.normal(0, 1.5, T) for _ in range(20)])
check("Lemma 3: variance <= 23^2 (1 - p)/p", var <= 23 ** 2 * (1 - p) / p, f"max sampled variance {var:.2f} vs bound {23 ** 2 * (1 - p) / p:.1f}")
zc = all(prog(gs(np.r_[R.normal(0, 2, 2), np.zeros(T - 2)], R.random() < p, p), 0) <= 3 for _ in range(2000))
check("probability-p zero-chain: a query never reveals more than one new coordinate", zc)


def sgd_queries(T, p, seed, h=.05, kmax=400000):
    Rr = np.random.default_rng(seed); x = np.zeros(T)
    for k in range(kmax):
        if prog(x, 0) >= T:
            return k
        x = x - h * gs(x, Rr.random() < p, p)
    return kmax


res = {}
for p_ in (1.0, .2, .05):
    for T_ in (4, 8, 16):
        res[(p_, T_)] = float(np.mean([sgd_queries(T_, p_, s) for s in range(20)]))
        print(f"  p = {p_:4.2f}, T = {T_:2d}: mean queries until all T coordinates are non-zero = {res[(p_, T_)]:.1f}  (T/p = {T_ / p_:.0f})")
check("Lemma 1: queries to activate all T coordinates >= about (T - log(1/delta))/(2p), and grow like T/p",
      all(res[(p_, T_)] >= (T_ - np.log(10)) / (2 * p_) for p_ in (.2, .05) for T_ in (8, 16)) and res[(.05, 16)] > 3 * res[(.2, 16)])
# path for the 3-D tab (T = 2): exact GD vs the noisy oracle
save_json("arjevani2023lower", "data.json", {"res": [[p_, T_, round(v, 2)] for (p_, T_), v in res.items()]})
done()
