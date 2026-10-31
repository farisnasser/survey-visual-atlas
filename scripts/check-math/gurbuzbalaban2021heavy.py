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

# in-page notes (papers/gurbuzbalaban2021heavy/notes.md): notes Sec. 3 exact values and primer numbers
zq = np.linspace(0, 1, 200001); phz = np.exp(-zq ** 2 / 2) / np.sqrt(2 * np.pi)
check("notes Sec. 3: 4 int_0^1 (1-z^2) phi = 4 phi(1) ~ 0.967883 (quadrature)", abs(4 * np.trapezoid((1 - zq ** 2) * phz, zq) - .967883) < 1e-6 and abs(4 * phz[-1] - .967883) < 1e-6)
check("notes Sec. 3: F(x) = E(ax-y)^2/2 = (x^2+1)/2 (a, y ~ N(0,1) independent)", all(np.isclose(.5 * (xx ** 2 * 1 + 1), (xx ** 2 + 1) / 2) for xx in [0., .7, -2.]))
mk = [0.]
for _ in range(6): mk.append(2 * mk[-1] + 1)
check("notes Sec. 3: E x_k^2 = 2 E x_{k-1}^2 + 1 = 2^k - 1 from x0 = 0", mk == [2 ** k - 1 for k in range(7)])
rng_g = np.random.default_rng(5); xg = np.zeros(400000)
for _ in range(4):
    ag, yg = rng_g.standard_normal(xg.size), rng_g.standard_normal(xg.size); xg = (1 - ag ** 2) * xg + ag * yg
check("notes Sec. 3: Monte Carlo E x_4^2 ~ 15", abs(np.mean(xg ** 2) / 15 - 1) < .1, f"{np.mean(xg ** 2):.2f}")
check("primer idea 1: x - eta a(ax - y) = (1 - eta a^2) x + eta a y", np.isclose(.3 - .2 * 1.7 * (1.7 * .3 - .4), (1 - .2 * 1.7 ** 2) * .3 + .2 * 1.7 * .4))
h2p = lambda s_: .5 * (.2 ** s_ + 1.6 ** s_)
lo_, hi_ = 1., 2.
for _ in range(80): lo_, hi_ = (((lo_ + hi_) / 2, hi_) if h2p((lo_ + hi_) / 2) < 1 else (lo_, (lo_ + hi_) / 2))
check("primer: M in {0.2, 1.6}: E log|M| ~ -0.57, h(1) = 0.9, h(2) = E M^2 = 1.3, alpha ~ 1.35", abs(.5 * np.log(.32) + .57) < 5e-3 and np.isclose(h2p(1), .9) and np.isclose(h2p(2), 1.3) and abs(lo_ - 1.35) < 5e-3, f"alpha = {lo_:.4f}")
rng_t = np.random.default_rng(0); Mt = np.where(rng_t.random(2000000) < .5, .2, 1.6); xt, xsim = 0., np.empty(Mt.size)
for i_, m_ in enumerate(Mt): xt = m_ * xt + 1.; xsim[i_] = xt
xsim = xsim[1000:]; p100, p1000 = np.mean(xsim > 100), np.mean(xsim > 1000)
check("primer: simulation of x+ = Mx + 1: P(x>100) ~ 0.009, P(x>1000) ~ 0.0004, ratio ~ 22 ~ 10^1.35", abs(p100 - .009) < .001 and abs(p1000 - .0004) < .0001 and abs(p100 / p1000 - 22) < 4 and abs(10 ** 1.35 - 22.4) < .1, f"{p100:.5f}, {p1000:.5f}, ratio {p100 / p1000:.1f}")
tt_ = np.logspace(0, 8, 400001)                # Pareto tail P(|X|>t) = t^-1.5 for t >= 1: E|X|^p finite iff p < 1.5
mom = lambda p_: 1 + np.trapezoid(p_ * tt_ ** (p_ - 1) * tt_ ** -1.5, tt_)
check("primer: tail-integral moments: p = 1 finite (= 3), p = 2 diverges for alpha = 1.5", abs(mom(1.) - 3) < .01 and mom(2.) > 1e3)
done()
