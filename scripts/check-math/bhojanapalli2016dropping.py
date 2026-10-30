"""Bhojanapalli, Kyrillidis, Sanghavi 2016 (COLT): factored gradient descent (FGD) U+ = U - eta grad f(UU^T) U.
Paper's own toy (Sec. 3): f(uu^T) = ||uu^T - Y||_F^2, Y = a^2 v1 v1^T - b^2 v2 v2^T, rank-1 u in R^2 (v1 = e1, v2 = e2):
    g(u) = (u1^2 - a^2)^2 + 2 u1^2 u2^2 + (u2^2 + b^2)^2,   grad f(X) = 2 (X - Y), M = m = 2.
Step (8): eta = 1 / (16 (M ||X0||_2 + ||grad f(X0)||_2)).  Theorem 4.2: Dist(U+, U*)^2 <= rho Dist(U, U*)^2 (r = rank X*, Cor. 4.3),
rho = 1 - m sigma_r(X*) / (64 (M ||X*||_2 + ||grad f(X*)||_2)), under (A2): Dist(U0, U*) <= sigma_r(U*) / (100 kappa) * sigma_r(X*)/sigma_1(X*).
"""
from common import *

print("bhojanapalli2016dropping")
b = 1.0
Gf = lambda u, a: (u[0] ** 2 - a ** 2) ** 2 + 2 * u[0] ** 2 * u[1] ** 2 + (u[1] ** 2 + b ** 2) ** 2
gG = lambda u, a: np.array([4 * u[0] * (u[0] ** 2 - a ** 2) + 4 * u[0] * u[1] ** 2, 4 * u[1] * u[0] ** 2 + 4 * u[1] * (u[1] ** 2 + b ** 2)])
Y = lambda a: np.diag([a ** 2, -b ** 2])
for a in [1.5, 2., 3.]:
    check_grad(f"g(u) = ||uu^T - Y||^2, a = {a}", lambda u: Gf(u, a), lambda u: gG(u, a), rand_pts(20, -3, 3))
    check("g matches the matrix form", all(abs(Gf(u, a) - np.sum((np.outer(u, u) - Y(a)) ** 2)) < 1e-9 for u in rand_pts(10, -3, 3)))
a = 2.
check_hess_at("at u = 0 is diag(-4a^2, 4b^2): saddle created by the factorization", lambda u: Gf(u, a), [0, 0], [[-4 * a ** 2, 0], [0, 4 * b ** 2]], tol=2e-3)
check_hess_at("at u* = (a, 0) is diag(8a^2, 4a^2 + 4b^2)", lambda u: Gf(u, a), [a, 0], [[8 * a ** 2, 0], [0, 4 * a ** 2 + 4 * b ** 2]], tol=5e-3)
fgd = lambda u, a, eta: u - eta * 2 * (np.outer(u, u) - Y(a)) @ u          # paper's update (factor 2 of the true gradient absorbed in eta)
check("FGD direction = (1/2) * true gradient of g", np.allclose(2 * 2 * (np.outer([.3, .7], [.3, .7]) - Y(a)) @ np.array([.3, .7]), gG(np.array([.3, .7]), a)))
step8 = lambda u0, a: 1 / (16 * (2 * np.linalg.norm(np.outer(u0, u0), 2) + np.linalg.norm(2 * (np.outer(u0, u0) - Y(a)), 2)))
dist = lambda u, a: min(np.linalg.norm(u - [a, 0]), np.linalg.norm(u + [a, 0]))   # O(1) = {+1, -1}
res = {}
for a in [.8, 1.5, 2., 3.]:
    u0 = np.array([a / 2, b / 10])     # the paper's example iterate
    e8, ecrit = step8(u0, a), 1 / (2 * a ** 2)
    out = {}
    for name, eta in [("paper", e8), ("1/M", .5)]:
        u = u0.copy(); k = 0
        while k < 20000 and dist(u, a) > 1e-8 and np.linalg.norm(u) < 1e6:
            u = fgd(u, a, eta); k += 1
        out[name] = (k, dist(u, a), np.linalg.norm(u))
    res[a] = (e8, ecrit, out)
    print(f"  a = {a}: eta(8) = {e8:.5f}, local stability limit 1/(2a^2) = {ecrit:.4f}; from u0 = (a/2, b/10): paper step -> dist {out['paper'][1]:.1e} in {out['paper'][0]} steps; eta = 1/M = 0.5 -> {'diverges' if out['1/M'][2] >= 1e6 else 'dist %.1e after %d' % (out['1/M'][1], out['1/M'][0])}")
check("eta = 1/M = 1/2 is fine for a < 1 but diverges from the paper's start once a >= 1.5", res[.8][2]["1/M"][1] < 1e-8 and all(res[a][2]["1/M"][2] >= 1e6 for a in [1.5, 2., 3.]))
check("the paper's step (8) converges for every a", all(res[a][2]["paper"][1] <= 1e-8 for res_a in [0] for a in res))
check("local stability limit of a constant step is 1/(2a^2) = 1/(2||X*||): eigenvalue of the linearized map at u*", all(abs(np.max(np.abs(eig(np.eye(2) - eta / 2 * np.diag([8 * a ** 2, 4 * a ** 2 + 4 * b ** 2])))) - 1) < 1e-12 for a in [2.] for eta in [1 / (2 * a ** 2)]))
# Theorem 4.2 / Cor. 4.3 bound from a start satisfying (A2), a = 2
a = 2.; Xs = np.outer([a, 0], [a, 0]); rho = 1 - 2 * a ** 2 / (64 * (2 * a ** 2 + np.linalg.norm(2 * (Xs - Y(a)), 2)))
u0 = np.array([a, 0]) + np.array([.6, .8]) * (a / 100) * .99           # Dist(U0, U*) <= sigma_r(U*)/100 (kappa = 1, rank 1)
e8 = step8(u0, a); u = u0.copy(); ok = True; d0 = dist(u0, a) ** 2; hist = [d0]
for k in range(1, 401):
    u = fgd(u, a, e8); hist.append(dist(u, a) ** 2); ok &= hist[-1] <= rho ** k * d0 * (1 + 1e-9)
print(f"  a = 2: Theorem 4.2 factor rho = {rho:.6f}; measured per-step ratio {hist[200] / hist[199]:.6f}; eta(8) = {e8:.5f}")
check("Cor. 4.3 bound Dist^2 <= rho^k Dist0^2 holds for 400 steps from an (A2) start", ok)
check("measured contraction is much faster than the bound (bound is conservative)", hist[200] / hist[199] < rho)
done()
