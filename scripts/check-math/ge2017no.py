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
done()
