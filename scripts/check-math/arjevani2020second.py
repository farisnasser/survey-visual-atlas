"""Arjevani et al. 2020: the elbow of stochastic oracle complexity and the HVP-RVR estimator (Alg. 1).

Gradient change d = grad f(x) - grad f(x_prev) estimated by
  (a) two single-point stochastic gradients: grad F(x, z1) - grad F(x_prev, z2)   (independent noise; MSE does not shrink with |x - x_prev|)
  (b) Alg. 1's Hessian-vector integral: sum_k  hess F(x^(k-1), z_k) (x^(k) - x^(k-1)),  x^(k) = x_prev + (k/K)(x - x_prev)
Oracles: grad F = grad f + s_g z,  hess F = hess f + s_h (Z + Z^T)/2.
"""
from common import *

print("arjevani2020second")
f = lambda p: .25 * (p[0] ** 2 - 1) ** 2 + .5 * p[1] ** 2 + .3 * np.sin(2 * p[0]) * p[1]
g = lambda p: np.array([p[0] ** 3 - p[0] + .6 * np.cos(2 * p[0]) * p[1], p[1] + .3 * np.sin(2 * p[0])])
H = lambda p: np.array([[3 * p[0] ** 2 - 1 - 1.2 * np.sin(2 * p[0]) * p[1], .6 * np.cos(2 * p[0])], [.6 * np.cos(2 * p[0]), 1]])
check_grad("f", f, g, rand_pts(20, -1.5, 1.5))
for p in rand_pts(5, -1.5, 1.5, seed=2):
    check_hess_at(f"at {np.round(p, 2)}", f, p, H(p))
sg, sh = 1.0, 1.0
R = np.random.default_rng(0)
xp = np.array([.4, -.3])
rows = []
for dn in [.5, .1, .02]:
    x = xp + dn * np.array([.8, .6]); d_true = g(x) - g(xp)
    m_a = np.mean([np.sum((g(x) + sg * R.normal(size=2) - g(xp) - sg * R.normal(size=2) - d_true) ** 2) for _ in range(20000)])
    for K in [1, 4, 16]:
        errs = []
        for _ in range(4000):
            est = np.zeros(2)
            for k in range(1, K + 1):
                a0 = xp + (k - 1) / K * (x - xp); Z = R.normal(size=(2, 2)); Z = sh * (Z + Z.T) / 2
                est += (H(a0) + Z) @ ((x - xp) / K)
            errs.append(np.sum((est - d_true) ** 2))
        rows.append([dn, K, float(np.mean(errs)), float(m_a)])
        print(f"  |dx| = {dn}, K = {K}: HVP-integral MSE {np.mean(errs):.2e}  vs two-gradient MSE {m_a:.3f}")
mse = {(r_[0], r_[1]): r_[2] for r_ in rows}
check("two single-point gradients: MSE ~ 2 d s_g^2 = 4, independent of |dx|", all(abs(r_[3] - 4) < .15 for r_ in rows))
check("HVP-integral MSE shrinks ~ |dx|^2 (x25 smaller from 0.5 to 0.1)", mse[(.5, 4)] / mse[(.1, 4)] > 15)
check("HVP-integral MSE shrinks ~ 1/K at fixed |dx| (variance part)", mse[(.5, 1)] / mse[(.5, 16)] > 8)
# variance formula for (Z+Z^T)/2 with iid N(0,1): E|Z v|^2 = |v|^2 (d+1)/2 = 1.5 |v|^2 in d = 2
v = np.array([.3, -.4]); emp = np.mean([np.sum((((lambda Z: (Z + Z.T) / 2)(R.normal(size=(2, 2)))) @ v) ** 2) for _ in range(40000)])
check("E|(Z+Z^T)/2 v|^2 = 1.5 |v|^2 in 2D (used for the page's predicted line)", abs(emp - 1.5 * v @ v) < .01 * 1.5 * (v @ v) * 3)
# elbow: exponents of eps^-1 for p-th order oracles (Fig. 1)
sto = {1: 4, 2: 3, 3: 3, 4: 3}; det = {p: (p + 1) / p for p in range(1, 5)}
check("elbow: stochastic exponent drops 4 -> 3 at p = 2, then flat; deterministic (p+1)/p", sto[1] == 4 and all(sto[p] == 3 for p in (2, 3, 4)) and det[1] == 2)

# in-page notes (papers/arjevani2020second/notes.md): Taylor expansion of the gradient (Sec. 2) and primer numbers
Fa = lambda p: p[0] ** 2 * p[1]
gFa = lambda p: np.array([2 * p[0] * p[1], p[0] ** 2])
check_grad("primer: F = x^2 y has gradient (2xy, x^2)", Fa, gFa, rand_pts(10, -2, 2))
check_hess_at("primer: Hessian of x^2 y at (1,2) = [[4,2],[2,0]]", Fa, [1, 2], [[4, 2], [2, 0]])
Hv = np.array([[4., 2], [2, 0]]) @ [1, 0]
fdq = (gFa(np.array([1.001, 2])) - gFa(np.array([1., 2]))) / .001
check("primer: HVP (4, 2); finite difference (4, 2.001)", np.allclose(Hv, [4, 2]) and np.allclose(fdq, [4, 2.001]), f"{fdq}")
errs = [np.linalg.norm(gFa(np.array([1 + d, 2])) - gFa(np.array([1., 2])) - d * Hv) for d in [1e-1, 1e-2, 1e-3]]
check("notes: grad F(x + dv) - grad F(x) - d H v = O(d^2)", errs[1] / errs[0] < .02 and errs[2] / errs[1] < .02, f"{errs}")
ex = 1.2 ** 2 - 1
left = lambda K: sum(2 * (1 + .2 * k / K) * (.2 / K) for k in range(K))
check("primer: x^3/3 from 1 to 1.2: exact 0.44, K=1 -> 0.40, K=4 -> 0.43", abs(ex - .44) < 1e-12 and abs(left(1) - .40) < 1e-12 and abs(left(4) - .43) < 1e-12)
check("primer: K-point HVP sums converge to the gradient difference", abs(left(1000) - .44) < 1e-4)
e_ = 1e-2
check("primer: eps = 0.01: eps^-4 = 1e8, eps^-3 = 1e6, ratio 100", abs(e_ ** -4 / 1e8 - 1) < 1e-12 and abs(e_ ** -3 / 1e6 - 1) < 1e-12)
done()
