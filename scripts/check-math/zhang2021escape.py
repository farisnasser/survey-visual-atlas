"""Zhang & Li 2021: negative-curvature finding with gradients only (Algorithm 1), replicating the setup of Fig. 1.

f(x1, x2) = x1^4/16 - x1^2/2 + 9 x2^2/8, saddle at 0 with H = diag(-1, 9/4); minima (+-2, 0), f = -1.
Algorithm 1: y_t = y_{t-1} - eta |y_{t-1}|/r (grad f(x~ + r y_{t-1}/|y_{t-1}|) - grad f(x~)),  y_0 ~ Uniform(B(r)).
PGD: x_0 ~ Uniform(B(r)) around the saddle, then GD with step eta.  Paper: eta = 0.05, r = 0.1, M = 300 samples,
t_NCGD = 30 vs t_PGD = 90.  Equal-budget descent comparison (our own setup): Algorithm 1 for 30 steps, then a kick of
length r along the output direction and 60 GD steps; PGD runs 90 GD steps.
"""
from common import *
from common import r as rnd

print("zhang2021escape")
f = lambda p: p[0] ** 4 / 16 - p[0] ** 2 / 2 + 9 * p[1] ** 2 / 8
g = lambda p: np.array([p[0] ** 3 / 4 - p[0], 9 * p[1] / 4])
check_grad("f", f, g, rand_pts(20, -2.5, 2.5))
check_hess_at("at the saddle", f, np.zeros(2), np.diag([-1, 2.25]))
check("minima at (+-2, 0) with f = -1", np.allclose(g([2, 0]), 0) and f([2, 0]) == -1)
eta, r, M, T1, T2 = .05, .1, 300, 30, 90
R = np.random.default_rng(78)


def ball(n):
    th = R.uniform(0, 2 * np.pi, n); rad = r * np.sqrt(R.uniform(0, 1, n))
    return np.stack([rad * np.cos(th), rad * np.sin(th)], 1)


Y0, X0 = ball(M), ball(M)


def alg1(y, T):
    for _ in range(T):
        n = np.linalg.norm(y)
        y = y - eta * n / r * (g(r * y / n) - g(np.zeros(2)))
    return y


def gdrun(x, T):
    for _ in range(T):
        x = x - eta * g(x)
    return x


aligned = lambda v: abs(v[0]) / np.linalg.norm(v) > np.cos(np.radians(5))
A30 = np.array([alg1(y, T1) for y in Y0])
share_a = np.mean([aligned(y) for y in A30])
check("Algorithm 1: >= 90% of the 300 directions within 5 deg of +-e1 after 30 steps", share_a >= .9, f"{share_a:.1%}")
e = A30 / np.linalg.norm(A30, axis=1, keepdims=True)
check("Rayleigh quotient e^T H e <= -sqrt(rho eps)/4-type bound: mean e^T H e close to -1", np.mean(-e[:, 0] ** 2 + 2.25 * e[:, 1] ** 2) < -.95)
P90 = np.array([gdrun(x, T2) for x in X0])
dec_p = f([0, 0]) - np.array([f(p) for p in P90])
# Lemma 6 kick: length (1/4) sqrt(eps/rho) along +-e (sign with the lower f); rho = sup |dH/dx| on |x1| <= 2.5 is 1.5*2.5,
# eps = 16 |lambda_min|^2 / rho is the largest eps with e^T H e <= -sqrt(rho eps)/4 at the saddle  ->  kick = 1/rho
rho = 1.5 * 2.5
kick = .25 * np.sqrt(16 / rho / rho)
print(f"  Lemma 6 kick length 1/rho = {kick:.4f}")
D1 = np.array([gdrun(min((kick * v, -kick * v), key=f), T2 - T1) for v in e])
dec_a = f([0, 0]) - np.array([f(p) for p in D1])
fa, fp = np.mean(dec_a <= .9), np.mean(dec_p <= .9)
check("equal budget (90 gradient steps): far fewer Algorithm-1 paths stall (decrease <= 0.9)", fa < .05 and fp > .2, f"Alg 1 {fa:.1%}, PGD {fp:.1%}")
print(f"  PGD paths with decrease <= 0.9: {fp:.1%} (paper Fig. 1: over 40%); Alg 1: {fa:.1%} (paper: under 5%)")
# alignment over time for the chart: share of Alg-1 directions / PGD displacements within 5 deg of +-e1
sa, sp = [], []
Y, X = Y0.copy(), X0.copy()
for t in range(T2 + 1):
    sa.append(np.mean([aligned(y) for y in Y])); sp.append(np.mean([aligned(x) for x in X]))
    Y = np.array([alg1(y, 1) for y in Y]); X = X - eta * np.array([g(x) for x in X])
check("PGD iterates also align with e1 but many are still short of the minima", sp[T2] > .9 and np.mean(np.abs(P90[:, 0]) < 1.5) > .2, f"aligned {sp[T2]:.1%}, |x1| < 1.5: {np.mean(np.abs(P90[:, 0]) < 1.5):.1%}")
save_json("zhang2021escape", "data.json", {"eta": eta, "r": r, "y0": rnd(Y0), "x0": rnd(X0), "share_a": rnd(sa), "share_p": rnd(sp),
                                         "kick": round(float(kick), 6), "fa": round(float(fa), 4), "fp": round(float(fp), 4)})
# in-page notes (papers/zhang2021escape/notes.md): worked example f = -x^2/2 + 2y^2, H = diag(-1, 4), u_{t+1} = u_t - 0.2 H u_t
Hn = np.diag([-1., 4.]); u = np.array([.1, .1]); us = []
for _ in range(2):
    u = u - .2 * Hn @ u; us.append(u.copy())
check("notes example u1 = (0.12, 0.02), u2 = (0.144, 0.004)", np.allclose(us[0], [.12, .02]) and np.allclose(us[1], [.144, .004]))
check("notes example: direction aligns with (1, 0)", abs(us[1][0]) / np.linalg.norm(us[1]) > abs(us[0][0]) / np.linalg.norm(us[0]) > .7)
# Taylor remainder bound ||grad f(x+u) - grad f(x) - Hu|| <= rho/2 ||u||^2 (rho-Hessian Lipschitz); the paper's Eq. (6) states the looser rho||u||^2
R2 = np.random.default_rng(1)
gc = lambda p: np.array([-p[0] + p[0] ** 2 / 2, 4 * p[1]])      # f = -x^2/2 + 2y^2 + x^3/6, rho = 1
worst = 0
for _ in range(2000):
    x0, uu = R2.normal(size=2), R2.normal(size=2) * R2.uniform(0, 1)
    Hx = np.diag([-1 + x0[0], 4.])
    worst = max(worst, np.linalg.norm(gc(x0 + uu) - gc(x0) - Hx @ uu) / (.5 * uu @ uu))
check("gradient-difference error <= rho/2 ||u||^2 (f with rho = 1)", worst <= 1 + 1e-9, f"max ratio {worst:.4f}")
# primer idea 2: f = -x^2/2 + 2y^2 + x^3/6 at 0, u = (0.1, 0): difference (-0.095, 0), Hu = (-0.1, 0), error 0.005 = rho/2 * 0.1^2
dg = gc(np.array([.1, 0])) - gc(np.zeros(2))
check("primer: gradient difference (-0.095, 0), Hu = (-0.1, 0), error 0.005", np.allclose(dg, [-.095, 0]) and np.isclose(np.linalg.norm(dg - np.array([-.1, 0])), .005))
# primer idea 1: decrease along curvature -gamma: gamma s^2/2 - rho s^3/6 at s = gamma/rho is gamma^3/(3 rho^2) = eps^1.5/(3 sqrt(rho))
for rho_, eps_ in [(1., 1e-4), (2., 1e-2), (.5, 3e-3)]:
    gam = np.sqrt(rho_ * eps_); s_ = gam / rho_
    assert np.isclose(gam * s_ ** 2 / 2 - rho_ * s_ ** 3 / 6, gam ** 3 / (3 * rho_ ** 2)) and np.isclose(gam ** 3 / (3 * rho_ ** 2), eps_ ** 1.5 / (3 * np.sqrt(rho_)))
check("primer: cubic-model decrease gamma^3/(3 rho^2) = eps^1.5/(3 sqrt rho) at gamma = sqrt(rho eps)", True)
# primer idea 3: random unit vector has typical overlap 1/sqrt(d); steps ln(sqrt d)/ln(1.01) = 694 (d = 1e6), 1388 (d = 1e12)
st = [np.log(np.sqrt(dd)) / np.log(1.01) for dd in (1e6, 1e12)]
check("primer: ln(sqrt d)/ln(1.01) ~ 694 and ~ 1388", round(st[0]) == 694 and round(st[1]) == 1388, f"{st[0]:.1f}, {st[1]:.1f}")
vv = R2.normal(size=(400, 10000)); vv /= np.linalg.norm(vv, axis=1, keepdims=True)
check("primer: overlap of a random unit vector with e1 is of order 1/sqrt(d)", .3 < np.median(np.abs(vv[:, 0])) * 100 < 1.5, f"median |u1| sqrt(d) = {np.median(np.abs(vv[:, 0])) * 100:.3f}")
# primer idea 4: escapes <= 384 (f0 - f*) sqrt(rho/eps^3); eps^-1.5 * eps^-0.25 = eps^-1.75
check("primer: escape count x per-escape cost gives eps^-7/4", np.isclose(-1.5 - .25, -1.75))
done()
