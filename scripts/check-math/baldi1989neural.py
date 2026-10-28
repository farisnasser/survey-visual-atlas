"""Baldi & Hornik 1989: linear autoencoder, Sigma = diag(9, 1), one hidden unit."""
from common import *

print("baldi1989neural")
S = np.diag([9.0, 1.0])


def E_full(A, B):  # E||x - ABx||^2 for centred x with covariance S
    M = np.eye(2) - A @ B
    return float(np.trace(M.T @ M @ S))


E = lambda w: float(np.trace(S) - 2 * w @ S @ w + (w @ w) * (w @ S @ w))   # tied slice B = A^T
Eg = lambda w: -4 * S @ w + 2 * w * (w @ S @ w) + 2 * (w @ w) * (S @ w)
P = rand_pts(40, -1.6, 1.6)
check("tied formula = full E with B = A^T", all(abs(E(p) - E_full(p.reshape(2, 1), p.reshape(1, 2))) < 1e-10 for p in P))
check_grad("tied", E, Eg, P)
for name, w, val in [("origin", [0, 0], 10), ("+u1", [1, 0], 1), ("+u2", [0, 1], 9)]:
    w = np.array(w, float)
    print(f"  {name}: E={E(w):.3f} eig H={np.round(eig(fd_hess(E, w)), 3)}")
    check(f"{name} is critical with E={val}", np.linalg.norm(Eg(w)) < 1e-9 and abs(E(w) - val) < 1e-9)
check("u1 is a minimum", all(eig(fd_hess(E, np.array([1., 0]))) > 0))
ev = eig(fd_hess(E, np.array([0., 1.])))
check("u2 is a saddle", ev[0] < 0 < ev[1], str(np.round(ev, 3)))
check("origin is a max on the tied slice", all(eig(fd_hess(E, np.zeros(2))) < 0))
check("origin is a saddle in full (A,B) space",
      E_full(np.array([[.1], [0]]), np.array([[-.1, 0]])) > 10 > E_full(np.array([[.1], [0]]), np.array([[.1, 0]])))
th = np.linspace(0, np.pi, 50)
check("E(theta) = 1 + 8 sin^2(theta)", np.allclose([E(np.array([np.cos(a), np.sin(a)])) for a in th], 1 + 8 * np.sin(th) ** 2))
U = lambda p: 9 * (1 - p[0] * p[1]) ** 2 + 1
check("untied slice formula", all(abs(U(p) - E_full(np.array([[p[0]], [0]]), np.array([[p[1], 0]]))) < 1e-10 for p in P))
Ug = lambda p: np.array([-18 * p[1] * (1 - p[0] * p[1]), -18 * p[0] * (1 - p[0] * p[1])])
check_grad("untied", U, Ug, P)
ends = [E(gd(Eg, s, .01, 4000, clip=1.6)[-1]) for s in rand_pts(40, -1.5, 1.5, seed=1) if np.linalg.norm(s) > .05]
check("random starts all reach E=1 (tied)", max(ends) < 1 + 1e-4, f"max final {max(ends):.5f}")
ends = [U(gd(Ug, s, .01, 4000, clip=1.6)[-1]) for s in rand_pts(40, -1.5, 1.5, seed=4) if np.linalg.norm(s) > .05]
check("random starts all reach E=1 (untied)", max(ends) < 1 + 1e-4, f"max final {max(ends):.5f}")
done()
