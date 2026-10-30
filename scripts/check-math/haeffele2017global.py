"""Haeffele & Vidal 2017: positively homogeneous network + regularizer of the SAME degree -> local minima with a zero
subnetwork are global (Theorem 1); mismatched degrees create a spurious local minimum at 0 (Prop. 19 of arXiv 1506.07540).

matched : f(u, v) = 1/2 (y - uv)^2 + lam/2 (u^2 + v^2)        phi = uv (degree 2), theta = (u^2+v^2)/2 (degree 2)
mismatch: f(u, v) = 1/2 (y - uv)^2 + lam (|u| + |v|)           theta degree 1 < 2  -> origin is a local min
ReLU net: g(x) = sum_i a_i relu(w_i . [x, 1]),  objective 1/(2n) sum (g - y)^2 + lam/2 sum (a_i^2 + |w_i|^2)  (degree 2 / 2)
"""
from common import *

print("haeffele2017global")
y = 1.0
fm = lambda p, lam: .5 * (y - p[0] * p[1]) ** 2 + lam / 2 * (p[0] ** 2 + p[1] ** 2)
gm = lambda p, lam: np.array([-(y - p[0] * p[1]) * p[1] + lam * p[0], -(y - p[0] * p[1]) * p[0] + lam * p[1]])
for lam in [.3, 1.2]:
    check_grad(f"matched lam = {lam}", lambda p: fm(p, lam), lambda p: gm(p, lam), rand_pts(20, -1.6, 1.6))
lam = .3
H0 = check_hess_at("matched at 0 = [[lam, -y], [-y, lam]]", lambda p: fm(p, lam), [0, 0], [[lam, -y], [-y, lam]])
check("matched, lam < y: origin is a strict saddle, lambda_min = lam - y", abs(eig(H0)[0] - (lam - y)) < 1e-3, f"{eig(H0)[0]:.3f}")
a = np.sqrt(y - lam); fstar = lam * y - lam ** 2 / 2
check("matched global min at u = v = sqrt(y - lam), value lam*y - lam^2/2 (soft threshold)", abs(fm([a, a], lam) - fstar) < 1e-12 and np.linalg.norm(gm([a, a], lam)) < 1e-12, f"f* = {fstar}")
# all GD runs from random starts end at the global value (no spurious minima)
R = np.random.default_rng(0); ends = []
for _ in range(200):
    p = gd(lambda q: gm(q, lam), R.uniform(-1.4, 1.4, 2), .1, 4000)[-1]; ends.append(fm(p, lam))
check("matched: 200/200 GD runs reach f*", np.max(np.abs(np.array(ends) - fstar)) < 1e-6)
H1 = fd_hess(lambda p: fm(p, 1.2), [0, 0])
check("matched, lam >= y: origin (the zero subnetwork) is a local min and it is the global one (x* = 0)", eig(H1)[0] > 0 and fm([0, 0], 1.2) == .5)

# mismatch: prox-gradient (ISTA) is the real optimizer for the l1 term
lm = .15
fx = lambda p: .5 * (y - p[0] * p[1]) ** 2 + lm * (abs(p[0]) + abs(p[1]))
soft = lambda z, t: np.sign(z) * np.maximum(np.abs(z) - t, 0)
def ista(p, eta=.05, steps=6000):
    p = np.array(p, float)
    for _ in range(steps):
        r_ = y - p[0] * p[1]
        p = soft(p - eta * np.array([-r_ * p[1], -r_ * p[0]]), eta * lm)
    return p
# global min of the mismatch objective: u = v = a with a (1 - a^2) = lm on the larger root
aa = np.linspace(0, 1.5, 300001); vals = .5 * (1 - aa ** 2) ** 2 + 2 * lm * aa; i = np.argmin(vals)
ag, fg = aa[i], vals[i]
check("mismatch global min below f(0) = 1/2", fg < .5, f"a = {ag:.4f}, f = {fg:.4f}")
eps = np.linspace(1e-4, .05, 50)
ring = min(fx([e * np.cos(t), e * np.sin(t)]) for e in eps for t in np.linspace(0, 2 * np.pi, 361))
check("mismatch: origin is a (spurious) local min: f > f(0) on a punctured disc of radius .05", ring > .5, f"min on disc {ring:.6f}")
R = np.random.default_rng(1); st = [ista(R.uniform(-1.4, 1.4, 2)) for _ in range(400)]
n0 = sum(np.linalg.norm(p) < 1e-9 for p in st); ng = sum(abs(fx(p) - fg) < 1e-5 for p in st)
print(f"  mismatch, 400 ISTA runs from U[-1.4, 1.4]^2: {n0} stuck at 0, {ng} at the global min")
check("mismatch: a sizeable fraction of runs is trapped at the origin, the rest reach the global min", n0 >= 20 and n0 + ng == 400, f"{n0}/400")

# ReLU network experiment: Theorem 1 says every local min with an all-zero neuron is global (same value for every r)
xs = np.linspace(-1, 1, 10); ys = np.sin(2.5 * xs) + .3 * xs
X = np.stack([xs, np.ones_like(xs)], 1); n = len(xs); LAM = 2e-3


def obj(a, W):
    Z = X @ W.T; return .5 / n * np.sum((np.maximum(Z, 0) @ a - ys) ** 2) + LAM / 2 * (np.sum(a ** 2) + np.sum(W ** 2))


def grad(a, W):
    Z = X @ W.T; A = np.maximum(Z, 0); res = (A @ a - ys) / n
    ga = A.T @ res + LAM * a
    gW = ((res[:, None] * (Z > 0)) * a[None, :]).T @ X + LAM * W
    return ga, gW


a0 = np.random.default_rng(3).normal(size=4); W0 = np.random.default_rng(4).normal(size=(4, 2))
check_grad("ReLU objective (away from kinks)", lambda t: obj(t[:4], t[4:].reshape(4, 2)), lambda t: np.concatenate([grad(t[:4], t[4:].reshape(4, 2))[0], grad(t[:4], t[4:].reshape(4, 2))[1].ravel()]), [np.concatenate([a0, W0.ravel()])])


def train(r_, S, seed, steps=200000, eta=.2):
    """S runs in parallel. Returns objective, its change over the last 20k steps (ReLU kinks keep |grad| from vanishing), number of all-zero neurons, certificate max_|w|=1 |<res, relu(Xw)>|/n."""
    R = np.random.default_rng(seed); a = R.normal(size=(S, r_)) * .8; W = R.normal(size=(S, r_, 2)) * .8
    def G(a, W):
        Z = np.einsum("nk,srk->snr", X, W); A = np.maximum(Z, 0); res = (np.einsum("snr,sr->sn", A, a) - ys) / n
        ga = np.einsum("snr,sn->sr", A, res) + LAM * a
        gW = np.einsum("snr,nk->srk", res[:, :, None] * (Z > 0) * a[:, None, :], X) + LAM * W
        return ga, gW, res
    for k in range(steps):
        if k == steps - 20000: fprev = [obj(a[s], W[s]) for s in range(S)]
        ga, gW, res = G(a, W); a -= eta * ga; W -= eta * gW
    ga, gW, res = G(a, W)
    th = np.linspace(0, 2 * np.pi, 7201); U = np.stack([np.cos(th), np.sin(th)], 1)
    cert = np.max(np.abs(res @ np.maximum(X @ U.T, 0)), 1)            # n * res already divided by n
    out = []
    for s in range(S):
        f = obj(a[s], W[s]); gn = abs(f - fprev[s]); sz = np.sqrt(a[s] ** 2 + np.sum(W[s] ** 2, 1))
        out.append([r_, f, gn, int(np.sum(sz < 1e-4)), cert[s]])
    return out


rows = []
for r_ in [1, 2, 3, 4, 6, 8, 12]:
    rows += train(r_, 16, 100 * r_)
rows = np.array(rows)
zero = rows[rows[:, 3] > 0]; ok = zero[zero[:, 4] <= LAM * (1 + 1e-3)]; bad = zero[zero[:, 4] > LAM * (1 + 1e-3)]; nz = rows[rows[:, 3] == 0]
fstar = ok[:, 1].min()
print(f"  ReLU: objective change over the last 20k steps: max {rows[:, 2].max():.1e}; runs still drifting by > 1e-7: {int(np.sum(rows[:, 2] > 1e-7))}/{len(rows)}")
print(f"  zero neuron + certificate (local min): {len(ok)} runs, objective range [{ok[:, 1].min():.7f}, {ok[:, 1].max():.7f}]")
print(f"  zero neuron, certificate fails (a new neuron would descend): {len(bad)} runs, objective >= {bad[:, 1].min() if len(bad) else 0:.5f}")
print(f"  no zero neuron: {len(nz)} runs, objective range [{nz[:, 1].min():.5f}, {nz[:, 1].max():.5f}]")
for r_ in [1, 2, 3, 4, 6, 8, 12]:
    R_ = rows[rows[:, 0] == r_]; print(f"    r = {r_:2d}: objective min {R_[:, 1].min():.5f} max {R_[:, 1].max():.5f}, zero neuron {int(np.sum(R_[:, 3] > 0))}/16, certified {int(np.sum((R_[:, 3] > 0) & (R_[:, 4] <= LAM * 1.001)))}/16")
check("ReLU: objective has settled in every run (change < 1e-4 over the last 20k steps)", rows[:, 2].max() < 1e-4)
check("ReLU: every certified run (zero neuron and max |<res, relu(Xw)>|/n <= lam) has the same objective", len(ok) >= 5 and ok[:, 1].max() - ok[:, 1].min() < 1e-6)
check("ReLU: no run anywhere goes below that value (it is the global minimum)", rows[:, 1].min() > fstar - 1e-7)
check("ReLU: zero-neuron runs that fail the certificate sit higher (not local minima)", len(bad) == 0 or bad[:, 1].min() > fstar + 1e-4)
save_json("haeffele2017global", "data.json", {"relu": [[int(q[0]), float(f"{q[1]:.7g}"), int(q[3]), int(q[3] > 0 and q[4] <= LAM * 1.001)] for q in rows], "fstar": float(f"{fstar:.7g}"), "lam": LAM,
                                               "xs": r(xs, 4), "ys": r(ys, 4), "mis": {"a": float(ag), "f": float(fg), "trapped": int(n0)}})
done()
