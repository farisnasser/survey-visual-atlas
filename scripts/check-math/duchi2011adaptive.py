"""Duchi, Hazan, Singer 2011: adaptive subgradient methods (AdaGrad).

Diagonal AdaGrad: x_{t+1,i} = x_{t,i} - eta g_{t,i} / sqrt(sum_s g_{s,i}^2)   (Eq. 1 with diag(G_t)^{1/2}, projection omitted).
Corollary 1: with eta = D_inf / sqrt 2, R(T) <= sqrt(2) D_inf sum_i |g_{1:T,i}|_2 (= sqrt(2 d) D_inf inf_s sqrt(sum g^T diag(s)^-1 g)).
Sparse example (Sec. 1.3): feature i appears with probability p_i = min(1, c i^-alpha); E sum_i |g_{1:T,i}|_2 <= sum_i sqrt(p_i T)
= O(log d sqrt T) for alpha = 2, versus O(sqrt(d T)) for online gradient descent on the box.
3D toy: logistic regression with one dense and one rare (3%) feature.
"""
from common import *

print("duchi2011adaptive")
# sparse example scaling
for d in [10, 100, 1000, 10000]:
    p = np.minimum(1, np.arange(1, d + 1) ** -2.)
    s = np.sum(np.sqrt(p))
    print(f"    d = {d}: sum_i sqrt(p_i) = {s:.2f} (log d = {np.log(d):.2f}), sqrt(d) = {np.sqrt(d):.1f}")
check("alpha = 2: sum_i sqrt(p_i) grows like log d (harmonic sum)", abs(np.sum(np.arange(1, 10001) ** -1.) - (np.log(10000) + .5772)) < 1e-3)
# Monte Carlo of E sum_i |g_{1:T,i}|_2 for hinge-loss gradients with |g_i| = 1 when feature i is present
rng = np.random.default_rng(0)
d, T = 200, 2000; p = np.minimum(1, np.arange(1, d + 1) ** -2.)
Z = rng.random((T, d)) < p
mc = np.sqrt(Z.sum(0)).sum()
check("Jensen: E sum_i |g_{1:T,i}| <= sum_i sqrt(p_i T)", mc <= np.sum(np.sqrt(p * T)), f"MC {mc:.1f} <= {np.sum(np.sqrt(p * T)):.1f}; sqrt(dT) = {np.sqrt(d * T):.0f}")

# 3D toy: logistic regression, x1 dense (always +-1 scaled), x2 rare (present 3% of the time)
N = 4000
X = np.zeros((N, 2)); X[:, 0] = rng.choice([-1, 1], N) * rng.uniform(.5, 1.5, N)
mask = rng.random(N) < .03; X[mask, 1] = rng.choice([-1, 1], mask.sum()) * rng.uniform(.5, 1.5, mask.sum())
WST = np.array([1.0, 2.5])
Y = np.where(rng.random(N) < 1 / (1 + np.exp(-X @ WST)), 1., -1.)
Lf = lambda w: np.mean(np.log1p(np.exp(-Y * (X @ w))))
Gf = lambda w: -(X * (Y / (1 + np.exp(Y * (X @ w))))[:, None]).mean(0)
check_grad("logistic loss", Lf, Gf, rand_pts(4, -3, 3))
wbest = np.zeros(2)
for _ in range(20000): wbest = wbest - 2 * Gf(wbest)
Lopt = Lf(wbest)


def run(kind, eta, T=1500, seed=1):
    r = np.random.default_rng(seed); w = np.array([-2.5, -2.5]); Gs = np.zeros(2) + 1e-8; out = []
    for t in range(T):
        i = r.integers(N); g = -X[i] * Y[i] / (1 + np.exp(Y[i] * X[i] @ w))
        if kind == "sgd": w = w - eta / np.sqrt(t + 1) * g
        else: Gs = Gs + g * g; w = w - eta * g / np.sqrt(Gs)
        out.append(Lf(w) - Lopt)
    return np.array(out), w


res = {}
for kind, grid in [("sgd", [.3, 1, 3, 10]), ("ada", [.1, .3, 1, 3])]:
    best = min(grid, key=lambda e: np.mean([run(kind, e, seed=s)[0][-200:].mean() for s in range(3)]))
    curve, w = run(kind, best)
    res[kind] = (best, curve, w)
    print(f"    {kind}: best eta {best}, final excess loss {curve[-200:].mean():.4f}, w = {np.round(w, 2)} (optimum {np.round(wbest, 2)})")
check("AdaGrad reaches lower excess loss than tuned 1/sqrt(t) SGD on the rare-feature problem", res["ada"][1][-200:].mean() < res["sgd"][1][-200:].mean())
check("AdaGrad gets the rare-feature weight closer to optimal", abs(res["ada"][2][1] - wbest[1]) < abs(res["sgd"][2][1] - wbest[1]), f"|w2 - w2*|: AdaGrad {abs(res['ada'][2][1] - wbest[1]):.2f}, SGD {abs(res['sgd'][2][1] - wbest[1]):.2f}")
save_json("duchi2011adaptive", "data.json", {"etaS": res["sgd"][0], "etaA": res["ada"][0], "wbest": r(wbest, 4),
                                             "X": r(X, 3), "Y": Y.astype(int).tolist(), "Lopt": round(float(Lopt), 6),
                                             "sgd": sig(res["sgd"][1][::10], 4), "ada": sig(res["ada"][1][::10], 4)})

# in-page notes (papers/duchi2011adaptive/notes.md): Sec. 3 proximal step and the primer's numbers
Rd = np.random.default_rng(50)
for _ in range(5):
    gt = Rd.normal(size=3); Ht = np.diag(Rd.uniform(.5, 3, 3)); xt = Rd.normal(size=3); et = .3
    obj = lambda x: gt @ (x - xt) + (x - xt) @ Ht @ (x - xt) / (2 * et)
    xs = xt - et * np.linalg.solve(Ht, gt)
    assert np.allclose(fd_grad(obj, xs), 0, atol=1e-5)
check("notes Sec. 3: argmin of the proximal surrogate is x_t - eta H_t^{-1} g_t", True)
check("primer idea 1: average regret sqrt(T)/T = 0.01 at T = 10000", np.isclose(np.sqrt(1e4) / 1e4, .01))
check("primer idea 2: every g in [-1,1] is a subgradient of |x| at 0",
      all(abs(y) >= gg * y for gg in np.linspace(-1, 1, 21) for y in np.linspace(-3, 3, 61)) and any(abs(y) < 1.1 * y for y in [1.0]))
T_ = 10000; G_busy, G_rare = T_ * 1.0, .01 * T_ * 1.0
check("primer idea 3: G = 10000 vs 100 -> effective steps eta/100 vs eta/10", G_busy == 1e4 and G_rare == 100 and np.isclose(1 / np.sqrt(G_rare) / (1 / np.sqrt(G_busy)), 10))
Hd = {d: sum(1 / j for j in range(1, d + 1)) for d in (10, 10 ** 4)}
check("primer idea 4: sum_j sqrt(p_j) = H_d: 2.93 (d=10), 9.79 (d=1e4); sqrt d = 3.16, 100",
      round(Hd[10], 2) == 2.93 and round(Hd[10 ** 4], 2) == 9.79 and round(np.sqrt(10), 2) == 3.16)
# Monte Carlo: E sum_j sqrt(sum_t g_tj^2) ~ sqrt(T) sum_j sqrt(p_j) for p_j = 1/j^2 (g_tj in {0, +-1})
d_, T2 = 50, 4000
pj = np.minimum(1, 1 / np.arange(1, d_ + 1) ** 2)
act = Rd.random((T2, d_)) < pj
emp = np.sum(np.sqrt(act.sum(0)))
check("primer idea 4: E sum_j sqrt(sum_t g^2) ~ sqrt(T) sum_j sqrt(p_j) (Jensen upper bound, close for frequent features)",
      emp <= np.sqrt(T2) * np.sum(np.sqrt(pj)) * 1.02 and emp >= .8 * np.sqrt(T2) * np.sum(np.sqrt(pj)), f"{emp:.1f} vs {np.sqrt(T2) * np.sum(np.sqrt(pj)):.1f}")
done()
