"""Carmon & Duchi 2019: gradient descent on the cubic-regularized quadratic f(x) = 1/2 x^T A x + b^T x + (rho/3)|x|^3.

GD (Eq. 9) from x0 = -r b/|b| (Assumption B, here r = 0); Lemma 2.3: |x_t| non-decreasing; Prop. 2.5: x_t -> global min if b^(1) != 0;
hard case b^(1) = 0: GD stays orthogonal to v1; Theorem 3.2: perturb b~ = b + sigma q.
"""
from common import *

print("carmon2019gradient")
lam = np.array([-.5, .6]); rho = 1.0
f = lambda x, b: .5 * x @ (lam * x) + b @ x + rho / 3 * np.linalg.norm(x) ** 3
g = lambda x, b: lam * x + b + rho * np.linalg.norm(x) * x
b0 = np.array([.06, -.5])
check_grad("f", lambda x: f(x, b0), lambda x: g(x, b0), rand_pts(20, -1.5, 1.5))


def stationary(b):
    """All stationary points: x = -b/(lam + rho r) with |x| = r, plus the hard-case circle when b1 = 0."""
    pts = []
    rs = np.linspace(1e-6, 4, 400001)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.sqrt(((b / (lam[None, :] + rho * rs[:, None])) ** 2).sum(1)) - rs
    ph = lambda rr: np.sqrt(((b / (lam + rho * rr)) ** 2).sum()) - rr
    for i in np.where(np.sign(phi[:-1]) != np.sign(phi[1:]))[0]:
        lo, hi = rs[i], rs[i + 1]
        if np.any(np.sign(lam + rho * lo) != np.sign(lam + rho * hi)):
            continue  # sign change through a pole, not a root
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.sign(ph(mid)) == np.sign(ph(lo)) else (lo, mid)
        pts.append(-b / (lam + rho * (lo + hi) / 2))
    if abs(b[0]) < 1e-15:
        rr = -lam[0] / rho; x2 = -b[1] / (lam[1] + rho * rr)
        if x2 ** 2 < rr ** 2:
            for s in (1, -1):
                pts.append(np.array([s * np.sqrt(rr ** 2 - x2 ** 2), x2]))
    return pts


def gd(b, eta, T, x0=None):
    x = np.zeros(2) if x0 is None else x0.copy(); xs = [x.copy()]
    for _ in range(T):
        x = x - eta * g(x, b); xs.append(x.copy())
    return np.array(xs)


eta = .1
for b in [np.array([.06, -.5]), np.array([.0, -.5]), np.array([.01, .3])]:
    S = stationary(b)
    vals = [f(s, b) for s in S]; xs_ = S[int(np.argmin(vals))]
    for s in S:
        check(f"b={b}: stationary point {np.round(s, 4)}", np.linalg.norm(g(s, b)) < 1e-5)
    X = gd(b, eta, 3000); nr = np.linalg.norm(X, axis=1)
    check(f"b={b}: |x_t| non-decreasing (Lemma 2.3)", np.all(np.diff(nr) >= -1e-12))
    if abs(b[0]) > 0:
        check(f"b={b}: GD converges to the global minimum (Prop. 2.5)", np.linalg.norm(X[-1] - xs_) < 1e-6, f"f* = {min(vals):.5f}")
    else:
        check("hard case b1 = 0: GD stays on the axis x1 = 0 and stops at a saddle", np.all(X[:, 0] == 0) and min(np.linalg.eigvalsh(np.diag(lam) + rho * (np.linalg.norm(X[-1]) * np.eye(2) + np.outer(X[-1], X[-1]) / np.linalg.norm(X[-1])))) < 0)
        bt = b + 1e-3 * np.array([.6, .8])
        Xp = gd(bt, eta, 3000)
        check("perturbed b~ = b + sigma q: GD reaches a global minimizer of the original f (up to sigma)", f(Xp[-1], b) - min(vals) < 1e-3, f"f = {f(Xp[-1], b):.5f} vs f* = {min(vals):.5f}")
# notes' 1D example
x = .2
for want in (.24, .2856):
    x = x - .25 * (-x + abs(x) * x)
    check(f"notes 1D: GD iterate {want}", abs(x - want) < 1e-12)

# Fig.-2-like experiment with our constants: d = 1000, lam_1 = -0.2, others in [-0.18, 1], b ~ [0.01, 1, 1, ...]
d = 1000; L = np.concatenate([[-.2], np.linspace(-.18, 1, d - 1)]); rho2, eta2, T = .2, .1, 6000
out = {}
for bn in [1.0, .3, .15, .05, 1e-3]:
    b = np.concatenate([[.01], np.ones(d - 1)]); b *= bn / np.linalg.norm(b)
    # global min via the secular equation: |b/(L + rho r)| = r, r > 0.2/rho
    lo, hi = .2 / rho2 + 1e-12, 50
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.linalg.norm(b / (L + rho2 * mid)) > mid else (lo, mid)
    xs_ = -b / (L + rho2 * hi); fs = .5 * xs_ @ (L * xs_) + b @ xs_ + rho2 / 3 * np.linalg.norm(xs_) ** 3
    x = np.zeros(d); gap, nr = [], []
    for t in range(T):
        if t % 20 == 0:
            gap.append(.5 * x @ (L * x) + b @ x + rho2 / 3 * np.linalg.norm(x) ** 3 - fs); nr.append(np.linalg.norm(x))
        x = x - eta2 * (L * x + b + rho2 * np.linalg.norm(x) * x)
    gap = np.maximum(np.array(gap), 1e-16)
    ups = int((np.diff(gap) > 1e-14).sum())
    print(f"  |b| = {bn}: |x*| = {hi:.3f}, gap after {T}: {gap[-1]:.2e}, f increases on {ups} of {len(gap) - 1} samples, |x_t| monotone {bool(np.all(np.diff(nr) >= -1e-12))}")
    check(f"|b| = {bn}: |x_t| non-decreasing", np.all(np.diff(nr) >= -1e-12))
    out[str(bn)] = {"gap": sig(gap, 3), "norm": sig(nr, 4), "rstar": round(float(hi), 4)}
check("a small-|b| instance stalls (gap > 1e-3 for > 1000 iterations) before converging", any(np.sum(np.array(v["gap"]) > 1e-3) > 50 for v in out.values()))
save_json("carmon2019gradient", "data.json", {"every": 20, "T": T, "rho": rho2, "eta": eta2, "runs": out})
done()
