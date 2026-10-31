"""Nesterov & Polyak 2006: cubic regularization of Newton's method.

T_M(x) = argmin_y <g, y-x> + 1/2 <H(y-x), y-x> + (M/6)|y-x|^3  (global minimizer, Eq. 2.4), x_{k+1} = T_M(x_k).
Toy f(x, y) = x^2/2 - y^2/2 + y^4/4 + 0.3 x y^2: strict saddle at 0, minima at (-0.3 y*^2, +-y*), y*^2 = 1/0.82.
Checks: subproblem solved globally; the cubic model is an upper bound of f when M >= L (Lemma 1, Eq. 2.3);
pure Newton is attracted by the saddle while CR leaves it (Lemma 6); f(x_k) monotone; min |grad| over the run.
"""
from common import *

print("nesterov2006cubic")


def cubic_step(g, H, M):
    lam, V = np.linalg.eigh(H); gb = V.T @ g
    lo = max(0, -2 * lam[0] / M) + 1e-14
    nr = lambda r: np.linalg.norm(gb / (lam + M * r / 2))
    if nr(lo) < lo:  # hard case
        Db = np.where(lam + M * lo / 2 > 1e-12, -gb / (lam + M * lo / 2), 0); Db[0] = np.sqrt(max(lo ** 2 - Db @ Db, 0))
        return V @ Db
    hi = lo + 1
    while nr(hi) > hi:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if nr(mid) > mid else (lo, mid)
    return V @ (-gb / (lam + M * hi / 2))


C3 = .3
f = lambda p: .5 * p[0] ** 2 - .5 * p[1] ** 2 + .25 * p[1] ** 4 + C3 * p[0] * p[1] ** 2
g = lambda p: np.array([p[0] + C3 * p[1] ** 2, -p[1] + p[1] ** 3 + 2 * C3 * p[0] * p[1]])
H = lambda p: np.array([[1, 2 * C3 * p[1]], [2 * C3 * p[1], -1 + 3 * p[1] ** 2 + 2 * C3 * p[0]]])
check_grad("f", f, g, rand_pts(20, -1.6, 1.6))
for p in rand_pts(4, -1.6, 1.6, seed=2):
    check_hess_at(f"at {np.round(p, 2)}", f, p, H(p), tol=5e-3)
ys = np.sqrt(1 / (1 - 2 * C3 * C3))
xm = -C3 * ys ** 2
check("minima at (-0.3 y*^2, +-y*)", np.linalg.norm(g([xm, ys])) < 1e-12 and eig(H([xm, ys]))[0] > 0, f"({xm:.4f}, {ys:.4f}), f = {f([xm, ys]):.4f}")
check("origin is a strict saddle", np.allclose(eig(H([0, 0])), [-1, 1]))

R = np.random.default_rng(0)
for t in range(6):
    gg = R.normal(size=2) * (0 if t == 0 else 1); A = R.normal(size=(2, 2)); HH = (A + A.T) / 2; M = R.uniform(.5, 3)
    m = lambda s: gg @ s + .5 * s @ HH @ s + M / 6 * np.linalg.norm(s) ** 3
    s = cubic_step(gg, HH, M)
    th = np.linspace(0, 2 * np.pi, 361); rr = np.linspace(0, 6, 601)
    scan = min(m(r_ * np.array([np.cos(a), np.sin(a)])) for a in th for r_ in rr[::2])
    check(f"cubic subproblem solved globally (instance {t})", m(s) <= scan + 1e-6, f"m(s) = {m(s):.5f}, scan {scan:.5f}")

# Hessian Lipschitz constant on the box |x|,|y| <= 1.6: dH = [[0, .6 dy], [.6 dy, 6 y dy + .6 dx]] -> sup operator norm
B = 1.6
Ls = []
for _ in range(20000):
    p, q = R.uniform(-B, B, 2), R.uniform(-B, B, 2)
    Ls.append(np.linalg.norm(H(p) - H(q), 2) / np.linalg.norm(p - q))
Lhat = max(Ls)
L = 10.0
check("Hessian-Lipschitz constant on the box is below L = 10", Lhat < L, f"sampled sup {Lhat:.3f}")

# Lemma 1 (Eq. 2.3): f(y) <= f(x) + <g, y-x> + 1/2 <H(y-x), y-x> + (L/6)|y-x|^3 for x, y in the box
x0 = np.array([1.2, .03])
model = lambda y, x, M: f(x) + g(x) @ (y - x) + .5 * (y - x) @ H(x) @ (y - x) + M / 6 * np.linalg.norm(y - x) ** 3
viol = max(f(y) - model(y, x0, L) for y in rand_pts(4000, -B, B, seed=5))
check("cubic model with M = L is an upper bound of f on the box", viol <= 1e-9, f"max f - model = {viol:.3f}")
viol0 = max(f(y) - model(y, x0, 0) for y in rand_pts(4000, -B, B, seed=5))
check("quadratic model (M = 0) is not an upper bound", viol0 > .1, f"max f - model = {viol0:.3f}")
s0 = cubic_step(g(x0), H(x0), L)
print(f"  first CR step from {x0}: -> {np.round(x0 + s0, 4)}, |s| = {np.linalg.norm(s0):.4f}")

# Pure Newton vs CR vs GD from x0 (near the saddle along the positive-curvature axis)
xn = x0.copy()
for _ in range(30):
    xn = xn - np.linalg.solve(H(xn), g(xn))
check("pure Newton converges to the saddle", np.linalg.norm(xn) < 1e-8, f"x = {xn}")


def run_cr(x, M, K=40):
    F, G = [f(x)], [np.linalg.norm(g(x))]; P = [x.copy()]
    for _ in range(K):
        x = x + cubic_step(g(x), H(x), M); F.append(f(x)); G.append(np.linalg.norm(g(x))); P.append(x.copy())
    return np.array(F), np.array(G), np.array(P)


F, G, P = run_cr(x0, L)
check("CR is monotone (Eq. 3.3 with M = L)", np.all(np.diff(F) <= 1e-12))
check("CR leaves the saddle and reaches a minimum", abs(abs(P[-1][1]) - ys) < 1e-6 and abs(P[-1][0] - xm) < 1e-6, f"x = {np.round(P[-1], 5)}")
kcr = int(np.argmax(G < 1e-6))
L1hat = max(np.linalg.norm(H(p), 2) for p in rand_pts(20000, -B, B, seed=6))
L1 = 8.0
check("gradient-Lipschitz constant on the box is below L1 = 8", L1hat < L1, f"sampled sup {L1hat:.3f}")
gdP = gd(g, x0, 1 / L1, 600)
check("GD stays in the box", np.abs(gdP).max() <= B)
kgd = int(np.argmax([np.linalg.norm(g(p)) < 1e-6 for p in gdP]))
check("GD (step 1/L1) also escapes but needs many more iterations", kgd > 3 * kcr, f"|grad| < 1e-6 after CR {kcr}, GD {kgd}")
# Theorem 1 with M_k = L_0 = L: min_{1<=i<=k} mu_L(x_i) <= (8/3) (3 (f0 - f*) / (2 k L))^(1/3), mu_L >= sqrt(|grad|/L)
D = f(x0) - f([xm, ys])
k = np.arange(1, 41)
bcr = L * ((8 / 3) * (3 * D / (2 * k * L)) ** (1 / 3)) ** 2
mcr = np.minimum.accumulate(G[1:41])
check("Theorem 1 bound holds on the CR run", np.all(mcr <= bcr), f"bound at k=1: {bcr[0]:.3f}, measured {mcr[0]:.3f}")
# gradient method with h = 1/L1: min_{0<=i<=k} |grad f(x_i)| <= sqrt(2 L1 (f0 - f*) / (k+1))  (Nesterov 2004, Eq. 1.2.13)
kk = np.arange(1, 601)
bgd = np.sqrt(2 * L1 * D / (kk + 1))
mgd = np.minimum.accumulate([np.linalg.norm(g(p)) for p in gdP])[1:]
check("gradient-method bound holds on the GD run", np.all(mgd <= bgd))
print(f"  f0 - f* = {D:.4f}")
# Lemma 6 at the exact saddle: the first CR step leaves along the negative eigenvector with length 2|lambda_min|/M
s = cubic_step(np.zeros(2), H([0, 0]), L)
check("at the exact saddle the cubic step has length 2|lambda_min|/M", abs(np.linalg.norm(s) - 2 / L) < 1e-9 and abs(s[0]) < 1e-12, f"s = {np.round(s, 4)}")
gx = [np.linalg.norm(g(p)) for p in gdP]
save_json("nesterov2006cubic", "data.json", {"L": L, "L1": L1, "D": round(float(D), 6), "x0": x0.tolist(), "cr_grad": sig(np.maximum(G, 1e-16)), "gd_grad": sig(np.maximum(gx[:300], 1e-16)), "kcr": kcr, "kgd": kgd})
# in-page notes (papers/nesterov2006cubic/notes.md): Sec. 3 first-order condition g + Hs + (M/2)|s| s = 0 at the model minimizer
Rn = np.random.default_rng(8)
for _ in range(5):
    A = Rn.normal(size=(3, 3)); Hm = (A + A.T) / 2; gm = Rn.normal(size=3); Mm = 2.
    mfun = lambda s: gm @ s + .5 * s @ Hm @ s + Mm / 6 * np.linalg.norm(s) ** 3
    # global minimizer via the 1-D secular equation r = |(H + M r/2 I)^{-1} g| with H + M r/2 I > 0 (Nesterov-Polyak Sec. 5)
    lo = max(0., -2 * eig(Hm)[0] / Mm) + 1e-12; hi = lo + 100
    phi = lambda r: np.linalg.norm(np.linalg.solve(Hm + Mm * r / 2 * np.eye(3), gm)) - r
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if phi(mid) > 0 else (lo, mid)
    s = -np.linalg.solve(Hm + Mm * lo / 2 * np.eye(3), gm)
    assert np.allclose(gm + Hm @ s + Mm / 2 * np.linalg.norm(s) * s, 0, atol=1e-8)
    assert all(mfun(s) <= mfun(s + .3 * Rn.normal(size=3)) for _ in range(200))
check("notes: the model's global minimizer solves g + Hs + (M/2)|s| s = 0 and (H + M|s|/2 I) s = -g", True)
# primer: Newton jumps to the saddle of x^2 - y^2; cubic Taylor cap tight for x^3/6; cubic step at g = 0, H = -1, M = 1
p0 = np.array([1., .5]); gN = np.array([2 * p0[0], -2 * p0[1]]); HN = np.diag([2., -2.]); sN = -np.linalg.solve(HN, gN)
check("primer: Newton from (1, 0.5) on x^2 - y^2: g = (2, -1), s = (-1, -0.5), lands on the saddle; f rises along y",
      np.allclose(gN, [2, -1]) and np.allclose(sN, [-1, -.5]) and np.allclose(p0 + sN, 0) and -(p0[1] + sN[1]) ** 2 > -p0[1] ** 2)
check("primer: f = x^3/6 at 0: f(2) = 8/6 ~ 1.33 equals the cap M|s|^3/6 with M = 1", np.isclose(2 ** 3 / 6, 1.333, atol=1e-3) and np.isclose(2 ** 3 / 6, 1 / 6 * 2 ** 3))
ss = np.linspace(-4, 4, 800001); mc = -.5 * ss ** 2 + abs(ss) ** 3 / 6
check("primer: m(s) = -s^2/2 + |s|^3/6 is minimized at |s| = 2 = 2|H|/M with value -2/3", np.isclose(abs(ss[np.argmin(mc)]), 2, atol=1e-4) and np.isclose(mc.min(), -2 / 3, atol=1e-8))
done()
