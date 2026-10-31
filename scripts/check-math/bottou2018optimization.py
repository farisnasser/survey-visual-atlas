"""Bottou, Curtis, Nocedal 2018 (SIAM Review 60(2); arXiv 1606.04838), Sec. 4.3, nonconvex objectives.
Theorem 4.8 (fixed step a <= mu / (L M_G)):  E[(1/K) sum_k |grad F(w_k)|^2] <= a L M / mu + 2 (F(w_1) - F_inf) / (K mu a)  -> a L M / mu.
Theorem 4.10 (diminishing steps, sum a = inf, sum a^2 < inf): E[sum a_k |grad F|^2] < inf, so the a-weighted average -> 0.
Toy: F(x, y) = (x^2 + y^2)/4 + sin x sin y (nonconvex, many minima), g = grad F + N(0, s^2 I): mu = mu_G = 1, M = 2 s^2, M_V = 0, M_G = 1.
"""
from common import *

print("bottou2018optimization")
F = lambda p: (p[..., 0] ** 2 + p[..., 1] ** 2) / 4 + np.sin(p[..., 0]) * np.sin(p[..., 1])
G = lambda p: np.stack([p[..., 0] / 2 + np.cos(p[..., 0]) * np.sin(p[..., 1]), p[..., 1] / 2 + np.sin(p[..., 0]) * np.cos(p[..., 1])], -1)
check_grad("F", lambda p: float(F(np.asarray(p))), lambda p: G(np.asarray(p)), rand_pts(20, -4, 4))
X = rand_pts(20000, -8, 8, seed=2)
lam = np.abs(.5 - np.sin(X[:, 0]) * np.sin(X[:, 1])) + np.abs(np.cos(X[:, 0]) * np.cos(X[:, 1]))
L = 1.5
check("gradient Lipschitz constant L = 1.5 (Hessian eigenvalues 1/2 - sin x sin y +- |cos x cos y|)", lam.max() <= L + 1e-12 and lam.max() > 1.49)
check("nonconvex: some Hessian eigenvalue is negative", np.min(.5 - np.sin(X[:, 0]) * np.sin(X[:, 1]) - np.abs(np.cos(X[:, 0]) * np.cos(X[:, 1]))) < -.4)
Finf = -1.0
check("F >= -1 = F_inf", F(X).min() >= Finf)
S = 1.0; M = 2 * S ** 2; w1 = np.array([2.5, -2.0]); F1 = float(F(w1))
Ks = np.unique(np.round(np.logspace(0, np.log10(3000), 40)).astype(int))
curves = {}
for a in [.05, .2, .5]:
    R = np.random.default_rng(1); W = np.tile(w1, (2000, 1)); acc = np.zeros(2000); out = []; k = 0
    for K in Ks:
        while k < K:
            acc += np.sum(G(W) ** 2, 1); W = W - a * (G(W) + S * R.normal(size=W.shape)); k += 1
        out.append([int(K), float(acc.mean() / K), a * L * M + 2 * (F1 - Finf) / (K * a)])
    curves[str(a)] = out
    print(f"  fixed a = {a}: E avg |grad|^2 at K = 3000: {out[-1][1]:.4f}; bound (4.28b) {out[-1][2]:.4f}; limit a L M = {a * L * M:.2f}")
    check(f"a = {a}: Theorem 4.8 bound holds at every K", all(q[1] <= q[2] for q in out))
check("the noise floor grows with the step: avg |grad|^2 at K = 3000 increases with a", curves["0.05"][-1][1] < curves["0.2"][-1][1] < curves["0.5"][-1][1])
# diminishing a_k = 0.5 / (1 + k/50): a-weighted average of |grad|^2 keeps falling
R = np.random.default_rng(1); W = np.tile(w1, (2000, 1)); num = np.zeros(2000); den = 0.; out = []; k = 0
for K in Ks:
    while k < K:
        a = .5 / (1 + k / 50); num += a * np.sum(G(W) ** 2, 1); den += a; W = W - a * (G(W) + S * R.normal(size=W.shape)); k += 1
    out.append([int(K), float(num.mean() / den), float(np.mean(np.sum(G(W) ** 2, 1)))])
curves["dim"] = out
print(f"  diminishing a_k = 0.5/(1 + k/50): weighted avg at K = 100, 1000, 3000: {out[np.searchsorted(Ks, 100)][1]:.4f}, {out[np.searchsorted(Ks, 1000)][1]:.4f}, {out[-1][1]:.4f}")
print(f"  diminishing: last-iterate E|grad F(w_K)|^2 at K = 3000: {out[-1][2]:.4f}")
check("Theorem 4.10: the weighted average keeps decreasing (K >= 300) with diminishing steps", all(out[i + 1][1] <= out[i][1] for i in range(np.searchsorted(Ks, 300), len(out) - 1)))
check("Corollary 4.12: the last iterate's E|grad F|^2 goes toward 0, below the a = 0.05 running average", out[-1][2] < curves["0.05"][-1][1])
save_json("bottou2018optimization", "data.json", {"curves": {k_: [[q[0], float(f"{q[1]:.4g}")] + ([float(f"{q[2]:.4g}")] if len(q) > 2 else []) for q in v] for k_, v in curves.items()}, "L": L, "M": M, "F1": F1})

# in-page notes (papers/bottou2018optimization/notes.md): expected-descent inequality and telescoped bound, primer numbers
rngB = np.random.default_rng(19)
Fq = lambda w: .5 * (w @ np.diag([1., 3.]) @ w)            # L = 3
gF_ = lambda w: np.diag([1., 3.]) @ w
LB, sig2 = 3., .5
ok = True
for _ in range(200):
    w = rngB.normal(size=2); al = rngB.uniform(.01, 1 / LB)
    gs = gF_(w) + rngB.normal(0, np.sqrt(sig2 / 2), (20000, 2))    # E||g - grad||^2 = sig2
    lhs = np.mean([Fq(w - al * g) for g in gs[:4000]])
    rhs = Fq(w) - al * gF_(w) @ gF_(w) + LB * al ** 2 / 2 * (gF_(w) @ gF_(w) + sig2)
    rhs2 = Fq(w) - al / 2 * gF_(w) @ gF_(w) + LB * al ** 2 * sig2 / 2
    ok &= lhs <= rhs + .02 and rhs <= rhs2 + 1e-12
check("notes: E[F(w+)] <= F - a|grad|^2 + (L a^2/2)(|grad|^2 + s^2) <= F - (a/2)|grad|^2 + L a^2 s^2/2 for a <= 1/L", ok)
w, al, T_, gsq = np.array([2., -1.]), .05, 4000, []
D0 = Fq(w)
for k in range(T_):
    gsq.append(gF_(w) @ gF_(w))
    w = w - al * (gF_(w) + rngB.normal(0, np.sqrt(sig2 / 2), 2))
check("notes: (1/T) sum E|grad F|^2 <= 2 Delta/(a T) + L a s^2 (one SG run, F_inf = 0)", np.mean(gsq) <= 2 * D0 / (al * T_) + LB * al * sig2, f"{np.mean(gsq):.4f} <= {2 * D0 / (al * T_) + LB * al * sig2:.4f}")
# primer
gB = np.array([1., 5.])
check("primer: g in {1, 5}: E g = 3, E g^2 = 13 = 9 + 4", gB.mean() == 3 and (gB ** 2).mean() == 13 == 9 + gB.var())
aS = np.sqrt(2 / 1e4); bS = 2 * np.sqrt(2 / 1e4)
ag = np.linspace(1e-4, .2, 200001); bnd = 2 / (ag * 1e4) + ag
check("primer: Delta = L = s^2 = 1, T = 1e4: alpha* ~ 0.0141, bound ~ 0.0283", abs(ag[bnd.argmin()] - aS) < 1e-5 and abs(bnd.min() - bS) < 1e-6 and abs(aS - .0141) < 5e-5 and abs(bS - .0283) < 5e-5)
check("primer: 4x the steps halves the optimised bound", abs(2 * np.sqrt(2 / 4e4) / bS - .5) < 1e-12)
done()
