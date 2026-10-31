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

# in-page notes (papers/baldi1989neural/notes.md): scalar example, PCA example, Sec. 5 curve, primer numbers
Ls = lambda p: (1 - p[0] * p[1]) ** 2
gLs = lambda p: np.array([-2 * p[1] * (1 - p[0] * p[1]), -2 * p[0] * (1 - p[0] * p[1])])
check_grad("notes: L(a,b) = (1-ab)^2 gradient", Ls, gLs, rand_pts(20, -2, 2))
check("notes: L = 1 - 2ab + a^2 b^2", all(abs(Ls(p) - (1 - 2 * p[0] * p[1] + (p[0] * p[1]) ** 2)) < 1e-12 for p in rand_pts(10, -2, 2)))
check("notes: L(0,0) = 1, L = 0 on ab = 1", Ls([0, 0]) == 1 and abs(Ls([2, .5])) < 1e-15)
check("notes: L(e,e) < 1 < L(e,-e) for 0 < e < 1", all(Ls([e, e]) < 1 < Ls([e, -e]) for e in [.01, .3, .9]))
check_hess_at("notes: H(0,0) = [[0,-2],[-2,0]]", Ls, [0, 0], [[0, -2], [-2, 0]])
check("notes: eigenvalues of H(0,0) are 2 and -2", np.allclose(np.linalg.eigvalsh([[0, -2], [-2, 0]]), [-2, 2]))
u = np.array([1., 0]); xv = np.array([3., 2])
check("notes: u^T x = 3, uu^T = [[1,0],[0,0]], Px = (3,0)", u @ xv == 3 and np.array_equal(np.outer(u, u), [[1, 0], [0, 0]]) and np.array_equal(np.outer(u, u) @ xv, [3, 0]))
Xc = rng_data = np.random.default_rng(4).normal(size=(400, 2)) * [3, 1]
Xc = Xc - Xc.mean(0); Sg = Xc.T @ Xc / len(Xc)
for uu in [np.array([1., 0]), np.array([np.cos(.7), np.sin(.7)])]:
    lhs = np.mean(np.sum((Xc - Xc @ np.outer(uu, uu)) ** 2, 1))
    assert abs(lhs - (np.trace(Sg) - uu @ Sg @ uu)) < 1e-10
check("notes: (1/N) sum ||x - uu^T x||^2 = tr(Sigma) - u^T Sigma u", True)
S9 = np.diag([9., 1])
th = np.linspace(0, np.pi, 181)
E9 = np.array([np.trace(S9) - np.array([np.cos(t), np.sin(t)]) @ S9 @ np.array([np.cos(t), np.sin(t)]) for t in th])
check("notes: E(theta) = 10 - (9cos^2 + sin^2) = 1 + 8 sin^2 theta", np.allclose(E9, 1 + 8 * np.sin(th) ** 2))
check("notes: E(0) = 1, E(pi/2) = 9", abs(E9[0] - 1) < 1e-12 and abs(E9[90] - 9) < 1e-12)
Ca = np.array([[2., 1], [0, 3]]); Aa = np.random.default_rng(1).normal(size=(3, 2)); Ba = np.random.default_rng(2).normal(size=(2, 3))
check("notes: AB = (AC)(C^-1 B)", np.allclose(Aa @ Ba, (Aa @ Ca) @ (np.linalg.inv(Ca) @ Ba)))
# primer
S54 = np.array([[5., 4], [4, 5]]); u1 = np.array([1, 1]) / np.sqrt(2); u2 = np.array([1, -1]) / np.sqrt(2)
check("primer: [[5,4],[4,5]] = 9 u1u1^T + 1 u2u2^T", np.allclose(S54, 9 * np.outer(u1, u1) + np.outer(u2, u2)))
c = np.array([u1 @ [1, 0], u2 @ [1, 0]])
check("primer: u = (1,0): c1^2 = c2^2 = 1/2 and u^T Sigma u = 5", np.allclose(c ** 2, .5) and np.array([1, 0]) @ S54 @ np.array([1, 0]) == 5)
check("primer: tr = 5 + 5 = 9 + 1 = 10", np.trace(S54) == 10 == np.linalg.eigvalsh(S54).sum())
check("primer: keeping u1 loses 1, keeping u2 loses 9", abs(10 - u1 @ S54 @ u1 - 1) < 1e-12 and abs(10 - u2 @ S54 @ u2 - 9) < 1e-12)
bb = np.linspace(-1, 2, 3001)
check("primer: (1 - 2b)^2 is minimised at b = 1/2", abs(bb[np.argmin((1 - 2 * bb) ** 2)] - .5) < 1e-9)
done()
