"""Gurbuzbalaban, Simsekli, Zhu 2021: heavy tails of streaming least-squares SGD with Gaussian data.

x_k = M_k x_{k-1} + q_k,  M_k = I - (eta/b) sum_i a_i a_i^T,  a_i ~ N(0, sigma^2 I_d)   (Theorem 3 setting)
h(s) = E ||M e1||^s,  tail index alpha solves h(alpha) = 1;  h(2) = 1 - 2 eta sigma^2 + eta^2 sigma^4 (d+b+1)/b.
Trick used here and in the page: M e1 = e1 - (eta/b) w with w = S e1, so ||M e1||^2 = 1 - 2(eta/b) w_1 + (eta/b)^2 ||w||^2
for every eta from one set of samples (w_1, ||w||^2).
"""
from common import *

print("gurbuzbalaban2021heavy")
rng = np.random.default_rng(0)


def wsamples(d, b, n, sigma=1.0):
    A = sigma * rng.normal(size=(n, b, d))
    w = np.einsum("nbi,nb->ni", A, A[:, :, 0])  # S e1 = sum_i a_i (a_i . e1)
    return w[:, 0], (w ** 2).sum(1)


def logh(s, eta, b, w1, ww):
    q = 1 - 2 * (eta / b) * w1 + (eta / b) ** 2 * ww
    lq = .5 * np.log(np.maximum(q, 1e-300))
    m = (s * lq).max()
    return m + np.log(np.mean(np.exp(s * lq - m))), lq.mean()


def alpha(eta, b, w1, ww, smax=12):
    lh, rho = logh(1, eta, b, w1, ww)
    if rho >= 0:
        return np.nan  # no contraction: no stationary law
    lo, hi = 1e-3, smax
    if logh(hi, eta, b, w1, ww)[0] < 0:
        return smax
    for _ in range(60):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if logh(mid, eta, b, w1, ww)[0] < 0 else (lo, mid)
    return (lo + hi) / 2


# scalar example from the notes: d = b = 1, eta = 1, sigma = 1
w1, ww = wsamples(1, 1, 400000)
q = np.sqrt(1 - 2 * w1 + ww)   # = |1 - a^2|
from math import erf, exp, pi, sqrt
check("E|1 - a^2| = 4 phi(1) ~ 0.96788", abs(q.mean() - 4 * exp(-.5) / sqrt(2 * pi)) < 5e-3, f"MC {q.mean():.5f}")
check("E (1 - a^2)^2 = 2", abs((q ** 2).mean() - 2) < .03, f"MC {(q ** 2).mean():.4f}")
a1 = alpha(1, 1, w1, ww)
check("scalar example: 1 < alpha < 2", 1 < a1 < 2, f"alpha = {a1:.3f}")
check("eta_crit = 2b/(sigma^2(d+b+1)) = 2/3 here", abs(2 * 1 / (1 + 1 + 1) - 2 / 3) < 1e-12)

# h(2) closed form vs MC for a few settings
for d, b, eta in [(5, 2, .2), (20, 5, .1), (1, 1, .5)]:
    w1, ww = wsamples(d, b, 200000)
    mc = np.exp(logh(2, eta, b, w1, ww)[0])
    cf = 1 - 2 * eta + eta ** 2 * (d + b + 1) / b
    check(f"h(2) closed form d={d} b={b} eta={eta}", abs(mc - cf) / cf < .02, f"MC {mc:.4f} vs {cf:.4f}")

# alpha = 2 exactly at eta_crit (Prop. 5)
d, b = 5, 2
w1, ww = wsamples(d, b, 300000)
ec = 2 * b / (d + b + 1)
ac = alpha(ec, b, w1, ww)
check("alpha(eta_crit) = 2", abs(ac - 2) < .05, f"alpha = {ac:.3f} at eta_crit = {ec:.3f}")

# curves for the page: alpha vs eta for b in {1,2,4,8} at d in {1,5,20}
curves = {}
for d in [1, 5, 20]:
    for b in [1, 2, 4, 8]:
        w1, ww = wsamples(d, b, 120000)
        etas = np.linspace(.02, 2.2 * 2 * b / (d + b + 1), 40)
        al = [alpha(e, b, w1, ww) for e in etas]
        curves[f"{d}-{b}"] = [[round(float(e), 4), (None if np.isnan(a) else round(float(a), 3))] for e, a in zip(etas, al)]
# Theorem 4 monotonicity spot checks
d = 5
for b1, b2 in [(1, 2), (2, 4), (4, 8)]:
    w1a, wwa = wsamples(d, b1, 150000); w1b, wwb = wsamples(d, b2, 150000)
    e = 1.15 * 2 * b1 / (d + b1 + 1)
    A1, A2 = alpha(e, b1, w1a, wwa), alpha(e, b2, w1b, wwb)
    check(f"alpha increases with batch {b1}->{b2} (eta = {e:.3f})", A1 < A2, f"{A1:.3f} < {A2:.3f}")
w1, ww = wsamples(5, 2, 150000)
check("alpha decreases with eta", alpha(.2, 2, w1, ww) > alpha(.4, 2, w1, ww) > alpha(.6, 2, w1, ww))
w5 = wsamples(5, 2, 150000); w20 = wsamples(20, 2, 150000)
A5, A20 = alpha(.16, 2, *w5), alpha(.16, 2, *w20)
check("alpha decreases with dimension (eta = 0.16, b = 2)", A5 > A20, f"d=5: {A5:.3f} > d=20: {A20:.3f}")

# simulate the scalar chain and compare a Hill tail estimate with alpha
n, K = 20000, 3000
x = np.zeros(n)
for k in range(K):
    a = rng.normal(size=n); y = rng.normal(size=n)
    x = (1 - a * a) * x + a * y
xs = np.sort(np.abs(x))[::-1]
kk = 400
hill = 1 / np.mean(np.log(xs[:kk] / xs[kk]))
check("scalar chain: Hill estimate near alpha", abs(hill - a1) < .35, f"Hill {hill:.3f} vs alpha {a1:.3f}")
save_json("gurbuzbalaban2021heavy", "data.json", {"curves": curves, "scalar_alpha": round(float(a1), 3)})
done()
