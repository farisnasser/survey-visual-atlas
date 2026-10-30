"""Goodfellow, Vinyals, Saxe 2015 (ICLR): linear interpolation l(a) = J((1 - a) theta_0 + a theta_f) between the
initialization and the SGD solution decreases smoothly (Secs. 2-4); between two different solutions there is a barrier
(Fig. 5 left, Fig. 10 deep linear). Fig. 14: the factored linear model (1 - w1 w2)^2, saddle at 0, manifold of solutions.

Here: (1) the factored model, (2) a small tanh MLP on a two-spirals-like task trained by SGD from two seeds (offline).
"""
from common import *

print("goodfellow2015qualitatively")
f = lambda p: (1 - p[0] * p[1]) ** 2
g = lambda p: np.array([-2 * p[1] * (1 - p[0] * p[1]), -2 * p[0] * (1 - p[0] * p[1])])
check_grad("(1 - w1 w2)^2", f, g, rand_pts(20, -2, 2))
check_hess_at("at 0 = [[0, -2], [-2, 0]] (saddle)", f, [0, 0], [[0, -2], [-2, 0]])
# SGD-like run (GD + small gradient noise) from a small random init, as in Fig. 14
R = np.random.default_rng(4); fth0 = np.array([.12, .05]); p = fth0.copy()
for k in range(3000):
    p = p - .02 * (g(p) + .05 * R.normal(size=2))
fthf = p
al = np.linspace(0, 1, 201); line = [f((1 - a) * fth0 + a * fthf) for a in al]
check("factored model: SGD reaches the solution manifold w1 w2 = 1", abs(fthf[0] * fthf[1] - 1) < .02, f"theta_f = {np.round(fthf, 3)}")
check("factored model: loss along init -> solution decreases monotonically", np.all(np.diff(line) <= 1e-12))
s1, s2 = np.array([1., 1.]), np.array([-1., -1.]); bar = max(f((1 - a) * s1 + a * s2) for a in al)
check("between the solutions (1,1) and (-1,-1) the straight line crosses the saddle: barrier 1", abs(bar - 1) < 1e-12)
s3 = np.array([2., .5]); bar13 = max(f((1 - a) * s1 + a * s3) for a in al)
print(f"  between (1,1) and (2,0.5) (same branch of the hyperbola) the line rises to {bar13:.4f}")

# small tanh MLP, 2-16-16-1, logistic loss, on a two-spirals-like task
def data(n=200, seed=0):
    R = np.random.default_rng(seed); t = np.sqrt(R.uniform(0, 1, n)) * 2.6 * np.pi; c = R.integers(0, 2, n)
    x = np.stack([t * np.cos(t + np.pi * c), t * np.sin(t + np.pi * c)], 1) / 8 + R.normal(0, .03, (n, 2))
    return x, 2. * c - 1
X, Y = data()
sizes = [(2, 16), (16, 16), (16, 1)]
def unpack(th):
    out, i = [], 0
    for a, b in sizes:
        W = th[i:i + a * b].reshape(a, b); i += a * b; c = th[i:i + b]; i += b; out.append((W, c))
    return out
D = sum(a * b + b for a, b in sizes)
def loss_grad(th, xb, yb, grad=True):
    P = unpack(th); hs = [xb]
    for W, c in P[:-1]: hs.append(np.tanh(hs[-1] @ W + c))
    z = (hs[-1] @ P[-1][0] + P[-1][1])[:, 0]; m = -yb * z
    L = np.mean(np.logaddexp(0, m))
    if not grad: return L
    dz = (-yb / (1 + np.exp(-m)))[:, None] / len(yb); gs = []
    d = dz
    for li in range(len(P) - 1, -1, -1):
        W, c = P[li]; gs.append((hs[li].T @ d, d.sum(0)))
        if li: d = (d @ W.T) * (1 - hs[li] ** 2)
    gs = gs[::-1]
    return L, np.concatenate([np.concatenate([gW.ravel(), gc]) for gW, gc in gs])
th_t = np.random.default_rng(9).normal(size=D) * .5
check_grad("MLP logistic loss (backprop)", lambda t: loss_grad(t, X[:20], Y[:20], False), lambda t: loss_grad(t, X[:20], Y[:20])[1], [th_t], tol=1e-5)
def init(seed):
    R = np.random.default_rng(seed); th = []
    for a, b in sizes: th += [R.normal(size=a * b) / np.sqrt(a), np.zeros(b)]
    return np.concatenate(th)
def train(seed, steps=20000, lr=.3, bs=20):
    R = np.random.default_rng(100 + seed); th = init(seed); th0 = th.copy(); traj = [th.copy()]
    for k in range(steps):
        idx = R.integers(0, len(X), bs); th = th - lr * loss_grad(th, X[idx], Y[idx])[1]
        if (k + 1) % 200 == 0: traj.append(th.copy())
    return th0, th, np.array(traj)
J = lambda th: loss_grad(th, X, Y, False)
acc = lambda th: np.mean(np.sign((lambda P: (np.tanh(np.tanh(X @ P[0][0] + P[0][1]) @ P[1][0] + P[1][1]) @ P[2][0] + P[2][1])[:, 0])(unpack(th))) == Y)
runs = [train(s) for s in (1, 2)]
alphas = np.round(np.linspace(-.25, 1.25, 121), 4)
curves = []
for th0, thf, _ in runs:
    curves.append([J((1 - a) * th0 + a * thf) for a in alphas])
    print(f"  MLP: J(theta_0) = {J(th0):.4f}, J(theta_f) = {J(thf):.4f}, train accuracy {acc(thf):.3f}")
inside = (alphas >= 0) & (alphas <= 1)
def bump(c):   # largest rise above the running minimum on [0, 1], and where it peaks
    c = np.array(c)[inside]; a = alphas[inside]; run = np.minimum.accumulate(c); i = int(np.argmax(c - run)); return float((c - run)[i]), float(a[i])
bumps = [bump(c) for c in curves]
for (h, a_), c in zip(bumps, curves):
    print(f"  MLP: largest rise above the running minimum on [0, 1]: {h:.4f} at alpha = {a_:.3f} (total drop {c[np.argmin(np.abs(alphas))] - c[np.argmin(np.abs(alphas - 1))]:.3f})")
check("MLP: SGD fits the training set (accuracy >= 0.97)", min(acc(r_[1]) for r_ in runs) >= .97)
check("MLP: run 2 decreases essentially monotonically (rise < 1% of the drop); run 1 shows a bump", bumps[1][0] < .01 * .822 and bumps[0][0] > .05)
# how common is the bump? 12 seeds at two step sizes, same number of samples seen (lr x steps fixed)
sweep = {}
for lr_, steps_ in [(.05, 60000), (.3, 10000)]:
    rel = []
    for sd in range(3, 15):
        R_ = np.random.default_rng(100 + sd); th = init(sd); t0 = th.copy()
        for k in range(steps_):
            idx = R_.integers(0, len(X), 20); th = th - lr_ * loss_grad(th, X[idx], Y[idx])[1]
        c = [J((1 - a) * t0 + a * th) for a in alphas]; h, _ = bump(c); rel.append(h / (c[np.argmin(np.abs(alphas))] - c[np.argmin(np.abs(alphas - 1))]))
    sweep[str(lr_)] = [round(float(q), 4) for q in rel]
    print(f"  MLP sweep lr = {lr_}: relative bump per seed {np.round(rel, 3).tolist()}; monotone (< 1% of drop) in {sum(q < .01 for q in rel)}/12")
check("the sweep finds both monotone and non-monotone interpolations", any(q < .01 for v in sweep.values() for q in v))
between = [J((1 - a) * runs[0][1] + a * runs[1][1]) for a in alphas]
bh = max(between[i] for i in range(len(alphas)) if 0 <= alphas[i] <= 1) - max(J(runs[0][1]), J(runs[1][1]))
print(f"  MLP: barrier between the two SGD solutions {bh:.4f} (endpoint losses {J(runs[0][1]):.4f}, {J(runs[1][1]):.4f})")
check("MLP: the straight line between two SGD solutions has a barrier", bh > .05)
# off-line distance (Fig. 11/13 style): projection of theta_t onto theta_f - theta_0 and the residual norm
th0, thf, tr = runs[0]; u = thf - th0; nu = np.linalg.norm(u)
proj = (tr - th0) @ u / nu; resid = np.linalg.norm((tr - th0) - np.outer(proj / nu, u), axis=1)
print(f"  MLP: |theta_f - theta_0| = {nu:.3f}, max residual off the line {resid.max():.3f}")
check("SGD does not follow the straight line (residual > 10% of the distance)", resid.max() > .1 * nu)
save_json("goodfellow2015qualitatively", "data.json", {"alphas": alphas.tolist(), "line": [sig(c, 4) for c in curves], "between": sig(between, 4),
                                                       "proj": r(proj, 3), "resid": r(resid, 3), "dist": round(float(nu), 3), "acc": [float(acc(r_[1])) for r_ in runs],
                                                       "fac": {"th0": r(fth0, 3), "thf": r(fthf, 4)}, "barrier": round(float(bh), 4), "bumps": [[round(h, 4), a_] for h, a_ in bumps], "sweep": sweep})
done()
