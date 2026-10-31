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
# in-page notes (papers/dinh2017sharp/notes.md): Sec. 2 rescaling identity, Sec. 3 Hessian of (ab - 1)^2 at (c, 1/c)
Rd = np.random.default_rng(4)
W1, W2, X = Rd.normal(size=(6, 3)), Rd.normal(size=(2, 6)), Rd.normal(size=(3, 50))
relu = lambda z: np.maximum(z, 0)
check("notes: f(x; cW1, W2/c) = f(x; W1, W2) for c = 0.1, 7", all(np.allclose((W2 / c) @ relu(c * W1 @ X), W2 @ relu(W1 @ X)) for c in [.1, 7.]))
Lab = lambda p: (p[0] * p[1] - 1) ** 2
for c in [.5, 1., 3., 10.]:
    Hc = fd_hess(Lab, np.array([c, 1 / c]))
    assert np.allclose(Hc, 2 * np.array([[c ** -2, 1], [1, c ** 2]]), atol=1e-3 * max(1, c * c)), (c, Hc)
    assert np.allclose(eig(2 * np.array([[c ** -2, 1], [1, c ** 2]])), [0, 2 * (c * c + c ** -2)], atol=1e-9)
check("notes: Hessian at (c, 1/c) is 2[[c^-2, 1], [1, c^2]], eigenvalues 0 and 2(c^2 + c^-2)", True)
# primer: same product, curvature in renamed coordinates, eps-ball sharpness 0.021 vs 0.161
check("primer: (1,1), (2,0.5), (4,0.25) give the same w1 w2 relu(x)", all(np.allclose(b * relu(a * X[0]), relu(X[0])) for a, b in [(1, 1), (2, .5), (4, .25)]))
check("primer: theta^2 has curvature 2; (phi/10)^2 has curvature 0.02", np.isclose(fd_hess(lambda p: p[0] ** 2, np.array([.3]))[0, 0], 2, atol=1e-4) and np.isclose(fd_hess(lambda p: (p[0] / 10) ** 2, np.array([.3]))[0, 0], .02, atol=1e-5))
D = np.diag([3., .2]); Hm = np.array([[2., .5], [.5, 1.]]); fq = lambda p: .5 * p @ Hm @ p
check("primer: theta = D phi turns H into D^T H D", np.allclose(fd_hess(lambda q: fq(D @ q), np.array([.1, .2])), D.T @ Hm @ D, atol=1e-4))
ang = np.linspace(0, 2 * np.pi, 200001)
epsS = lambda a, b, e=.1: float(np.max(Lab(np.array([a + e * np.cos(ang), b + e * np.sin(ang)]))))
s1, s4 = epsS(1, 1), epsS(4, .25)
check("primer: eps = 0.1 ball sharpness ~0.021 at (1,1), ~0.161 at (4,0.25)", np.isclose(s1, .021, atol=5e-4) and np.isclose(s4, .161, atol=5e-4), f"{s1:.4f}, {s4:.4f}")
done()
