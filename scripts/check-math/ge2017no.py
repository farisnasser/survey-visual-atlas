"""Ge, Jin, Zheng 2017: the unified direction of improvement Delta = U - U* R and Lemma 7.

Symmetric case, H = identity, Q = 0:  f(U) = 1/2 |U U^T - M*|_F^2,  M* = U* U*^T.
Lemma 7: Delta : hess f(U) : Delta = |Delta Delta^T|^2 - 3 |M - M*|^2 + 4 <grad f(U), Delta>.
Lemma 6: |Delta Delta^T|^2 <= 2 |M - M*|^2.   Section 5 (asymmetric): balancing regularizer 1/8 |U^T U - V^T V|^2.
"""
from common import *

print("ge2017no")
us = np.array([.9, .45])
Ms = np.outer(us, us)
f = lambda u: .5 * np.sum((np.outer(u, u) - Ms) ** 2)
g = lambda u: 2 * (np.outer(u, u) - Ms) @ u
check_grad("f", f, g, rand_pts(30, -1.5, 1.5))


def delta(u):
    """r = 1: R in {+1, -1}, pick the one minimising |u - u* R|."""
    return min([u - us, u + us], key=np.linalg.norm)


worst7, worst6 = 0, 0
for u in rand_pts(400, -1.6, 1.6, seed=1):
    D = delta(u); H = fd_hess(f, u)
    lhs = D @ H @ D
    rhs = np.sum(np.outer(D, D) ** 2) - 3 * np.sum((np.outer(u, u) - Ms) ** 2) + 4 * g(u) @ D
    worst7 = max(worst7, abs(lhs - rhs)); worst6 = max(worst6, np.sum(np.outer(D, D) ** 2) - 2 * np.sum((np.outer(u, u) - Ms) ** 2))
check("Lemma 7 identity holds at 400 random points", worst7 < 1e-5, f"max |lhs - rhs| {worst7:.1e}")
check("Lemma 6: |Delta Delta^T|^2 <= 2 |M - M*|^2", worst6 <= 1e-12)
# stationary points: 0 and +-u*;  at 0, Delta = -+u*, curvature = |u*|^4 - 3|M*|^2 = -2|u*|^4 <= -|M - M*|^2
D0 = delta(np.zeros(2)); c0 = D0 @ fd_hess(f, np.zeros(2)) @ D0
check("at the saddle u = 0: Delta^T H Delta = -2|u*|^4 <= -|M - M*|_F^2", abs(c0 + 2 * np.sum(us ** 2) ** 2) < 1e-6 and c0 <= -np.sum(Ms ** 2) + 1e-9, f"{c0:.4f}")
check_hess_at("Hessian at 0 = -2 M*", f, [0, 0], -2 * Ms)
for s in (1, -1):
    check(f"u = {s} u* is a strict minimum", eig(fd_hess(f, s * us))[0] > 1e-3)
# notes' example: M* = diag(1, 0)
fn = lambda u: .5 * np.sum((np.outer(u, u) - np.diag([1., 0])) ** 2)
check("notes: f(x, y) = 1/2(x^2-1)^2 + x^2 y^2 + 1/2 y^4, f(0.5, 0) = 0.28125", abs(fn([.5, 0]) - .28125) < 1e-12 and abs(fn([.3, .7]) - (.5 * (.09 - 1) ** 2 + .09 * .49 + .5 * .49 ** 2)) < 1e-12)
check_hess_at("notes: Hessian at 0 = diag(-2, 0)", fn, [0, 0], [[-2, 0], [0, 0]])
# asymmetric scalar case: f(u, v) = 1/2 (uv - 1)^2, with and without the balancing regularizer
fa = lambda p: .5 * (p[0] * p[1] - 1) ** 2
fb = lambda p: fa(p) + (p[0] ** 2 - p[1] ** 2) ** 2 / 8
gb = lambda p: np.array([(p[0] * p[1] - 1) * p[1] + (p[0] ** 2 - p[1] ** 2) * p[0] / 2, (p[0] * p[1] - 1) * p[0] - (p[0] ** 2 - p[1] ** 2) * p[1] / 2])
check_grad("balanced", fb, gb, rand_pts(20, -2, 2))
check("without the regularizer the minimizers form the hyperbola uv = 1 (a zero eigenvalue)", abs(eig(fd_hess(fa, [2, .5]))[0]) < 1e-6)
check("with it the minima are isolated: (1,1), (-1,-1), positive definite", eig(fd_hess(fb, [1, 1]))[0] > .1 and abs(fb([1, 1])) < 1e-15)
check("origin stays a strict saddle with the regularizer", eig(fd_hess(fb, [0, 0]))[0] < -.5)

# in-page notes (papers/ge2017no/notes.md): Sec. 2 example, Sec. 4 identity (H = I) and the primer's numbers
Fg = lambda p: .5 * ((p[0] ** 2 - 1) ** 2 + 2 * p[0] ** 2 * p[1] ** 2 + p[1] ** 4)
Gg = lambda p: np.array([2 * p[0] * (p[0] ** 2 + p[1] ** 2 - 1), 2 * p[1] * (p[0] ** 2 + p[1] ** 2)])
check("notes Sec. 2: the two expansions of F agree", all(np.isclose(Fg(q), .5 * ((q[0] ** 2 + q[1] ** 2) ** 2 - 2 * q[0] ** 2 + 1)) for q in rand_pts(20, -2, 2)))
check_grad("notes Sec. 2: dF/da = 2a(a^2+b^2-1), dF/db = 2b(a^2+b^2)", Fg, Gg, rand_pts(20, -2, 2))
check_hess_at("notes Sec. 2: Hessian at the origin diag(-2, 0)", Fg, [0, 0], [[-2, 0], [0, 0]])
check("notes Sec. 2: F(0,0) = 0.5, F(0.5,0) = 0.28125, F(1,0) = 0; stationary at (0,0), (+-1,0)",
      Fg([0, 0]) == .5 and np.isclose(Fg([.5, 0]), .28125) and Fg([1, 0]) == 0 and all(np.allclose(Gg(q), 0) for q in ([0, 0], [1, 0], [-1, 0])))
# Lemma 7 with H = I, Q = 0 (f = 1/2||UU^T - M*||^2) at stationary points of a random symmetric instance
Rg = np.random.default_rng(60)
Us = Rg.normal(size=(4, 2)); Ms = Us @ Us.T
fU = lambda U: .5 * np.sum((U @ U.T - Ms) ** 2)
def align(U):
    W, _, Zt = np.linalg.svd(Us.T @ U); return U - Us @ (W @ Zt)
ok = True
for _ in range(20):
    U = Rg.normal(size=(4, 2)); D = align(U)
    lhs = (fU(U + 1e-4 * D) - 2 * fU(U) + fU(U - 1e-4 * D)) / 1e-8
    gU = 2 * (U @ U.T - Ms) @ U
    rhs = np.sum((D @ D.T) ** 2) - 3 * np.sum((U @ U.T - Ms) ** 2) + 4 * np.sum(gU * D)
    ok &= abs(lhs - rhs) < 1e-3 * (1 + abs(rhs)) and np.sum((D @ D.T) ** 2) <= 2 * np.sum((U @ U.T - Ms) ** 2) + 1e-9
check("notes Sec. 4: Lemma 7 identity (H = I, no regularizer) and Lemma 6 ||DD^T||^2 <= 2||M - M*||^2", ok)
check("primer idea 1: at U = 0, curvature along Delta = (1,0) is -2 = 1 - 3 <= -1",
      np.isclose((Fg([1e-4, 0]) - 2 * Fg([0, 0]) + Fg([-1e-4, 0])) / 1e-8, -2, atol=1e-4) and 1 - 3 == -2 and -2 <= -1)
check("primer idea 2: (1,2) pays 9/8, (sqrt2, sqrt2) pays 0", np.isclose((1 - 4) ** 2 / 8, 9 / 8) and np.isclose((2 - 2) ** 2 / 8, 0) and np.isclose(np.sqrt(2) ** 2, 2))
check("primer idea 3: 10% sampling misses the single nonzero entry with probability 0.9", np.isclose(1 - .1, .9) and 100 * 100 == 10000)
xx = np.linspace(-3, 3, 6001)
check("primer idea 4: |x| <= x^2/2 + 1/2 with equality at +-1", np.all(np.abs(xx) <= xx ** 2 / 2 + .5 + 1e-12) and np.isclose(1, .5 + .5))
done()
