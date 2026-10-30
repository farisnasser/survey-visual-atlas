"""Islamov, Ajroldi, Orvieto, Lucchi 2024: the alpha-beta condition (Definition 1, Eq. 2)
    <grad f_i(x), x - x_p> >= alpha (f_i(x) - f_i(x_p)) - beta (f_i(x) - f_i*)    for all i, some x_p in Proj(x, S),
allows saddle points, unlike PL. Paper's Example 2: f = (f1 + f2)/2, f1 = 1 - exp(-|x|^2), f2 = 1 - exp(-|x - (2,2)|^2),
claimed to satisfy Definition 1 with beta = alpha - 8 for some alpha. Theorem 1 (SGD, gamma <= (alpha - beta)/(2L)):
    min_k E[f(x_k) - f*] <= E dist(x0, S)^2 / (K gamma (alpha - beta)) + 2 L gamma s2/(alpha - beta) + 2 beta s2/(alpha - beta),
s2 = sigma_int^2 = E_i[f* - f_i*].
"""
from common import *

print("islamov2024loss")
c2 = np.array([2., 2.])
f1 = lambda p: 1 - np.exp(-np.sum(np.asarray(p) ** 2, -1))
f2 = lambda p: 1 - np.exp(-np.sum((np.asarray(p) - c2) ** 2, -1))
g1 = lambda p: 2 * np.asarray(p) * np.exp(-np.sum(np.asarray(p) ** 2))
g2 = lambda p: 2 * (np.asarray(p) - c2) * np.exp(-np.sum((np.asarray(p) - c2) ** 2))
f = lambda p: .5 * (f1(p) + f2(p))
g = lambda p: .5 * (g1(p) + g2(p))
check_grad("f = (f1 + f2)/2", f, g, rand_pts(30, -1.5, 3.5))
# minimizers: on the diagonal by symmetry; Newton on t for x = y = t
t = .01
for _ in range(60):
    h = 1e-6; d1 = (f([t + h] * 2) - f([t - h] * 2)) / (2 * h); d2 = (f([t + h] * 2) - 2 * f([t] * 2) + f([t - h] * 2)) / h ** 2; t -= d1 / d2
m1 = np.array([t, t]); m2 = c2 - m1; fstar = f(m1)
check("two global minimizers m1, m2 = (2,2) - m1, gradient zero there", np.linalg.norm(g(m1)) < 1e-9 and abs(f(m2) - fstar) < 1e-12, f"m1 = ({t:.6f}, {t:.6f}), f* = {fstar:.6f}")
Hm = fd_hess(f, m1); Hs = fd_hess(f, [1, 1])
check("m1 is a strict local min (Hessian PD)", eig(Hm)[0] > 0, f"eig {np.round(eig(Hm), 4)}")
check("(1, 1) is a strict saddle: gradient 0, one negative eigenvalue", np.linalg.norm(g([1, 1])) < 1e-12 and eig(Hs)[0] < 0 < eig(Hs)[1], f"eig {np.round(eig(Hs), 4)}, f = {f([1, 1]):.6f}")
# global minima: grid search confirms nothing lower
G = np.linspace(-3, 5, 801); XX, YY = np.meshgrid(G, G); P = np.stack([XX, YY], -1)
check("grid search: no value below f*", f(P).min() > fstar - 1e-9)
# PL fails: |grad f|^2 / (2 (f - f*)) = 0 at the saddle
check("PL fails at the saddle: |grad|^2 = 0 while f - f* > 0", np.linalg.norm(g([1, 1])) == 0 and f([1, 1]) - fstar > .1)
# alpha-beta with beta = alpha - 8 and fi* = 0:  <grad fi, x - xp> + alpha fi(xp) >= 8 fi(x)  for i = 1, 2
# -> required alpha(x) = max_i (8 fi(x) - <grad fi(x), x - xp>) / fi(xp), best over xp in Proj(x, S)
B = 1.5; Gb = np.linspace(-B, 2 + B, 241); req = np.zeros((len(Gb), len(Gb)))
F1 = lambda q: 1 - np.exp(-np.sum(q ** 2)); F2 = lambda q: 1 - np.exp(-np.sum((q - c2) ** 2))
for j, yv in enumerate(Gb):
    for i, xv in enumerate(Gb):
        x = np.array([xv, yv]); d1_, d2_ = np.linalg.norm(x - m1), np.linalg.norm(x - m2)
        cands = [m1] if d1_ < d2_ - 1e-12 else [m2] if d2_ < d1_ - 1e-12 else [m1, m2]
        best = np.inf
        for xp in cands:
            a_ = max((8 * F1(x) - g1(x) @ (x - xp)) / F1(xp), (8 * F2(x) - g2(x) @ (x - xp)) / F2(xp))
            best = min(best, a_)
        req[j, i] = best
amin = float(req.max())
print(f"  alpha needed on [-{B}, {2 + B}]^2 with beta = alpha - 8: alpha >= {amin:.2f} (worst point {Gb[np.unravel_index(req.argmax(), req.shape)[1]]:.2f}, {Gb[np.unravel_index(req.argmax(), req.shape)[0]]:.2f})")
check("Example 2 satisfies Definition 1 on the box for a finite alpha with beta = alpha - 8", np.isfinite(amin) and amin > 8)
# Theorem 1 constants: L = 2 (max |eig| of the Hessian of 1 - exp(-r^2) is 2 at r = 0), s2 = f* (f_i* = 0)
r_ = np.linspace(0, 4, 4001); lam = np.maximum(2 * np.exp(-r_ ** 2), np.abs((2 - 4 * r_ ** 2) * np.exp(-r_ ** 2)))
L = float(lam.max()); s2 = fstar
check("each f_i is L-smooth with L = 2", abs(L - 2) < 1e-12)
alpha = amin; beta = alpha - 8
floor = 2 * beta * s2 / (alpha - beta)
print(f"  Theorem 1 with alpha = {alpha:.2f}: step gamma <= {(alpha - beta) / (2 * L):.2f}, error floor 2 beta s2/(alpha - beta) = {floor:.3f} vs max(f - f*) = {1 - fstar:.3f}")
check("for this example the bound's floor exceeds the whole range of f - f* (the guarantee is vacuous here)", floor > 1 - fstar)

# any other split delta = alpha - beta gives the same picture: floor ~ 2 s2 / f_i(x_p) because f1(m1) ~ 1e-6
for dl in [1., 8., 100.]:
    A = max(max((dl * F1(np.array(x)) - g1(np.array(x)) @ (np.array(x) - xp)) / F1(xp), (dl * F2(np.array(x)) - g2(np.array(x)) @ (np.array(x) - xp)) / F2(xp)) for x in [[-1.5, 3.5], [1, 1], [3.5, -1.5]] for xp in [m1 if np.linalg.norm(np.array(x) - m1) <= np.linalg.norm(np.array(x) - m2) else m2])
    print(f"  alpha - beta = {dl:g}: alpha >= {A:.3g} at three test points, floor >= {2 * (A - dl) * s2 / dl:.3g}")
print(f"  f1 at its nearest global minimizer: {F1(m1):.3g}")
# SGD (pick i uniformly, constant step 0.5): where 400 runs from U[-1.5, 3.5]^2 end after 3000 steps
R = np.random.default_rng(0); X = R.uniform(-1.5, 3.5, (400, 2))
for k in range(3000):
    I = R.integers(0, 2, 400); C = np.where(I[:, None] == 0, 0., 2.); D = X - C
    X = X - .5 * 2 * D * np.exp(-np.sum(D ** 2, 1))[:, None]
dist = np.minimum(np.linalg.norm(X - m1, axis=1), np.linalg.norm(X - m2, axis=1))
print(f"  SGD step 0.5: {np.mean(dist < .05):.3f} of runs within 0.05 of a global minimizer, {np.mean(f(X) > .95):.3f} still on the far plateau (f > 0.95)")
check("SGD: most runs reach a global minimizer, none stay at the saddle", np.mean(dist < .05) > .8 and np.min(np.linalg.norm(X - 1, axis=1)) > .3)
diag = np.linspace(-1.5, 3.5, 201); pl = [float(np.sum(g([t_, t_]) ** 2) / (2 * (f([t_, t_]) - fstar))) if f([t_, t_]) - fstar > 1e-9 else None for t_ in diag]
save_json("islamov2024loss", "data.json", {"m1": r(m1, 6), "fstar": float(fstar), "alpha": round(amin, 2), "L": L, "floor": round(floor, 3),
                                           "req": {"dom": [-B, 2 + B], "z": r(np.log10(np.maximum(req[::4, ::4], 1e-3)), 3)},
                                           "pl": [[round(float(t_), 3), (None if v is None else float(f"{v:.4g}"))] for t_, v in zip(diag, pl)]})
done()
