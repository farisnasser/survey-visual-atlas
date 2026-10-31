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

# in-page notes (papers/carmon2019gradient/notes.md): notes Sec. 1 example in full, primer numbers
q1 = lambda s_: -s_ ** 2 / 2 + abs(s_) ** 3 / 3
check("notes Sec. 1: q'(s) = -s + s|s| has zeros 0, +-1; q(+-1) = -1/6; q''(0) = -1", all(abs(-s_ + s_ * abs(s_)) < 1e-12 for s_ in (0., 1., -1.)) and np.isclose(q1(1.), -1 / 6) and np.isclose((q1(1e-4) - 2 * q1(0) + q1(-1e-4)) / 1e-8, -1, atol=1e-3))
s1_ = .2 - .25 * (-.2 + .04); s2_ = s1_ - .25 * (-s1_ + s1_ * abs(s1_))
check("notes Sec. 1: s1 = 0.24, s2 = 0.2856 (eta = 0.25, s0 = 0.2), and (0.24)^2 = 0.0576", np.isclose(s1_, .24) and np.isclose(s2_, .2856) and np.isclose(.24 ** 2, .0576))
sx = .2
for _ in range(200): sx -= .25 * (-sx + sx * abs(sx))
check("notes Sec. 1: GD from 0.2 with eta = 0.25 converges to +1", abs(sx - 1) < 1e-9)
check("primer: grad of |s|^3/3 is |s| s (finite differences)", np.allclose([(np.linalg.norm(v + e) ** 3 - np.linalg.norm(v - e) ** 3) / 6e-6 for v in [np.array([.3, -.7])] for e in np.eye(2) * 1e-6], np.linalg.norm([.3, -.7]) * np.array([.3, -.7]), atol=1e-6))
A2 = np.diag([-1., 2.]); b2 = np.array([.1, .2]); rh = 1.
qq = lambda s_: b2 @ s_ + .5 * s_ @ A2 @ s_ + rh / 3 * np.linalg.norm(s_) ** 3
lams = np.linspace(-1.9999, 4, 600001); crit = []
fsec = np.array([rh * np.linalg.norm(b2 / np.array([-1 + l, 2 + l])) - l if abs(-1 + l) > 1e-9 else np.nan for l in lams])
for i_ in np.where(np.sign(fsec[:-1]) * np.sign(fsec[1:]) < 0)[0]:
    lo_, hi_ = lams[i_], lams[i_ + 1]
    if abs(lo_ - 1) < 1e-3 or abs(hi_ - 1) < 1e-3: continue
    for _ in range(80):
        m_ = (lo_ + hi_) / 2
        g_ = rh * np.linalg.norm(b2 / np.array([-1 + m_, 2 + m_])) - m_
        if np.sign(g_) == np.sign(rh * np.linalg.norm(b2 / np.array([-1 + lo_, 2 + lo_])) - lo_): lo_ = m_
        else: hi_ = m_
    if m_ > 0: crit.append((m_, -b2 / np.array([-1 + m_, 2 + m_])))
glob = [c for c in crit if c[0] >= 1]
print(f"  primer critical points (lambda, s): {[(round(l, 4), np.round(v, 4).tolist(), round(qq(v), 4)) for l, v in crit]}")
check("primer: exactly one critical point has shift >= 1: lambda* ~ 1.092, s* ~ (-1.090, -0.065), q ~ -0.278, and it is the lowest", len(glob) == 1 and abs(glob[0][0] - 1.092) < 1e-3 and np.allclose(glob[0][1], [-1.090, -.065], atol=1e-3) and abs(qq(glob[0][1]) + .278) < 1e-3 and all(qq(glob[0][1]) <= qq(v) for _, v in crit) and len(crit) >= 2)
beta_ = 2.; R_ = beta_ / (2 * rh) + np.sqrt((beta_ / (2 * rh)) ** 2 + np.linalg.norm(b2) / rh); eta_ = 1 / (4 * (beta_ + rh * R_))
xg_ = np.zeros(2)
for _ in range(5000): xg_ = xg_ - eta_ * (A2 @ xg_ + b2 + rh * np.linalg.norm(xg_) * xg_)
check("primer: R ~ 2.106, eta <= 0.061; GD from 0 reaches s*", abs(R_ - 2.106) < 1e-3 and abs(eta_ - .061) < 5e-4 and np.allclose(xg_, glob[0][1], atol=1e-6))
check("primer: 1-D example (beta = 1, b = 0): R = 1, Assumption A bound 0.125 < 0.25", np.isclose(.5 + np.sqrt(.25), 1) and np.isclose(1 / (4 * (1 + 1)), .125))
bh = np.array([0., .2]); xh = np.zeros(2)
for _ in range(5000): xh = xh - eta_ * (A2 @ xh + bh + rh * np.linalg.norm(xh) * xh)
qh = lambda s_: bh @ s_ + .5 * s_ @ A2 @ s_ + rh / 3 * np.linalg.norm(s_) ** 3
yh = -.2 / 3; sh = np.array([np.sqrt(1 - yh ** 2), yh])
check("primer: hard case b = (0, 0.2): GD stays on the axis at (0, -0.095), q ~ -0.0097; true min ~ -0.173 (lambda* = 1)", xh[0] == 0 and abs(xh[1] + .095) < 1e-3 and abs(qh(xh) + .0097) < 1e-4 and abs(qh(sh) + .173) < 1e-3 and np.allclose((A2 + np.eye(2)) @ sh, -bh) and np.isclose(np.linalg.norm(sh), 1))
done()
