"""Dinh, Pascanu, Bengio, Bengio 2017 (ICML; arXiv 1703.04933): sharp minima can generalize.
alpha-scale transformation T_a(theta1, theta2) = (a theta1, theta2 / a) leaves a ReLU network's function unchanged (Def. 5), but
Theorem 3: Hessian at T_a(theta) = D_a^-1 H D_a^-1-type rescaling; Theorem 4: its spectral norm can be made arbitrarily large;
Theorem 2: the eps-flat set C(L, theta, eps) has infinite volume.
Toy: y = theta2 * relu(theta1 * x), one datum x = 1, y = 1: L(t1, t2) = (t2 relu(t1) - 1)^2; minima t1 t2 = 1, t1 > 0.
"""
from common import *

print("dinh2017sharp")
L = lambda p: (p[1] * max(p[0], 0) - 1) ** 2
g = lambda p: np.array([2 * (p[1] * p[0] - 1) * p[1], 2 * (p[1] * p[0] - 1) * p[0]]) if p[0] > 0 else np.array([0., 2 * (p[1] * max(p[0], 0) - 1) * max(p[0], 0)])
check_grad("L for t1 > 0", L, g, rand_pts(20, .2, 2))
rows = []
for a in [.25, .5, 1, 2, 4]:
    th = np.array([a, 1 / a]); H = fd_hess(L, th); ev = eig(H)
    same = all(abs(1 / a * max(a * x, 0) - max(x, 0)) < 1e-12 for x in np.linspace(-2, 2, 9))
    sharp = max(L(th + .1 * np.array([np.cos(t), np.sin(t)])) for t in np.linspace(0, 2 * np.pi, 3601))   # eps-sharpness (Def. 2), eps = 0.1
    rows.append([a, float(ev[1]), float(ev[0]), sharp])
    check(f"a = {a}: same function, Hessian eigenvalues (0, 2(a^2 + 1/a^2))", same and abs(ev[1] - 2 * (a * a + 1 / a ** 2)) < 1e-3 and abs(ev[0]) < 1e-3, f"{np.round(ev, 4)}, eps-sharpness {sharp:.4f}")
check("sharpness grows without bound along the same valley (Theorem 4)", rows[-1][1] > 30 > rows[2][1])
# Theorem 3: H(T_a theta) = D H(theta) D with D = diag(1/a, a)
a = 3.; H1 = fd_hess(L, [1, 1]); Ha = fd_hess(L, [a, 1 / a]); D = np.diag([1 / a, a])
check("Theorem 3: H(T_a theta) = D_a H(theta) D_a, D_a = diag(1/a, a)", np.max(np.abs(Ha - D @ H1 @ D)) < 2e-3)
# Theorem 2: the eps-flat set contains the whole valley t1 t2 = 1, t1 > 0, which has infinite length
check("Theorem 2: every point of the hyperbola has L = 0 < eps, and the arc length to a = 1000 already exceeds 999", all(L([q, 1 / q]) < 1e-15 for q in np.logspace(-3, 3, 50)))
save_json("dinh2017sharp", "data.json", {"rows": [[r_[0], round(r_[1], 4), round(r_[3], 5)] for r_ in rows]})
done()
