"""Murty & Kabadi 1987: local-minimum testing is NP-hard; copositivity; subset sum as a box QP (Example 2)."""
from common import *

print("murty1987some")
rng = np.random.default_rng(0)
simplex = rng.dirichlet([1, 1], 20000)
for q in [-1.5, -1.0, -.6, 0, .5, 1.0, 1.5]:
    Q = np.array([[1, q], [q, 1]])
    mn = np.einsum("ni,ij,nj->n", simplex, Q, simplex).min()
    psd = eig(Q)[0] >= -1e-12
    check(f"q={q}: copositive iff q >= -1 (min on simplex {mn:.4f}); PSD iff |q| <= 1", (mn >= -1e-9) == (q >= -1) and psd == (abs(q) <= 1))
check("closed form min over simplex = (1+q)/2 for q <= 1", abs(np.einsum("ni,ij,nj->n", simplex, np.array([[1, -.6], [-.6, 1]]), simplex).min() - .2) < 1e-3)

# quartic p(u) = u1^4 + 2q u1^2 u2^2 + u2^4 = (u^2)^T Q (u^2)
for q in [-1.4, -1.0, -.5, .7]:
    p = lambda u, q=q: u[0] ** 4 + 2 * q * u[0] ** 2 * u[1] ** 2 + u[1] ** 4
    gp = lambda u, q=q: np.array([4 * u[0] ** 3 + 4 * q * u[0] * u[1] ** 2, 4 * u[1] ** 3 + 4 * q * u[0] ** 2 * u[1]])
    check_grad(f"quartic q={q}", p, gp, rand_pts(20, -1.2, 1.2))
    check_hess_at(f"quartic q={q} at 0 (zero: second-order test silent)", p, [0, 0], [[0, 0], [0, 0]], tol=1e-5)
    th = np.linspace(0, 2 * np.pi, 3601)
    ring = min(p([1e-2 * np.cos(t), 1e-2 * np.sin(t)]) for t in th)
    check(f"quartic q={q}: origin is a local min iff Q copositive", (ring >= -1e-18) == (q >= -1), f"min on small circle {ring:.2e}")
    path = gd(gp, [.6, .62], .05, 4000, clip=1.2)   # the page clamps to the drawn box [-1.2, 1.2]^2
    print(f"  q={q}: GD from (0.6, 0.62) ends at {np.round(path[-1], 3)}, p = {p(path[-1]):.2e}")
    if q < -1:
        check("q < -1: GD runs away along the diagonal to the edge of the box, p < 0", p(path[-1]) < -.1 and abs(path[-1][0]) > 1.1)
    elif q == -1:
        check("q = -1: p = (u1^2 - u2^2)^2, GD stops on the zero valley u1 = u2 (non-strict minima)", p(path[-1]) < 1e-10 and abs(path[-1][0] - path[-1][1]) < 1e-3)
    else:
        check(f"q >= -1: GD creeps into the degenerate minimum at 0", np.linalg.norm(path[-1]) < .1)

# Example 2: subset sum as a box-constrained QP
d = np.array([3., 5.])
F = lambda y, d0: (d @ y - d0) ** 2 + (y * (1 - y)).sum()
gF = lambda y, d0: 2 * (d @ y - d0) * d + (1 - 2 * y)
check_grad("subset-sum QP", lambda y: F(y, 4), lambda y: gF(y, 4), rand_pts(20, 0, 1))
G = np.stack(np.meshgrid(np.linspace(0, 1, 801), np.linspace(0, 1, 801)), -1).reshape(-1, 2)
res = {}
for d0 in [3, 4, 5, 8]:
    vals = ((G @ d - d0) ** 2 + (G * (1 - G)).sum(1))
    mn = vals.min(); at = G[vals.argmin()]
    # projected GD from a grid of starts
    ends = []
    for s in rand_pts(36, 0, 1, seed=d0):
        y = s.copy()
        for _ in range(3000):
            y = np.clip(y - .01 * gF(y, d0), 0, 1)
        ends.append(F(y, d0))
    ends = np.array(ends)
    res[d0] = (mn, at, ends)
    print(f"  d0={d0}: global min {mn:.4f} at {at}; projected GD lands at values {np.unique(np.round(ends, 3))}")
    check(f"d0={d0}: min 0 iff a subset of {{3,5}} sums to d0", (mn < 1e-9) == (d0 in (0, 3, 5, 8)))
check("d0 = 4: no subset -> minimum stays positive", res[4][0] > .1)
check("projected GD can land in non-global box minima (d0 = 5)", (res[5][2] > 1e-3).any() or (res[3][2] > 1e-3).any())

# in-page notes (papers/murty1987some/notes.md): q(x,y) = x^2 - y^2 teaching example (notes Secs. 1-2)
qn = lambda p: p[0] ** 2 - p[1] ** 2
check_hess_at("notes: q = x^2 - y^2 has Hessian diag(2, -2)", qn, [0, 0], [[2, 0], [0, -2]])
check("notes: grad q(0,0) = 0 and q(0,t) = -t^2 < 0", np.allclose([2 * 0, -2 * 0], 0) and all(qn([0, t]) == -t * t < 0 for t in [.1, -.5, 2]))
# primer: brute-force counts, Example 2 with d = (3, 5), copositive but indefinite Q
check("primer: 2^20 = 1 048 576 and 2^60 ~ 1.15e18", 2 ** 20 == 1048576 and abs(2 ** 60 / 1.15e18 - 1) < .003)
Gp = np.stack(np.meshgrid(np.linspace(0, 1, 4001), np.linspace(0, 1, 4001)), -1).reshape(-1, 2)
v8 = (Gp @ d - 8) ** 2 + (Gp * (1 - Gp)).sum(1)
v4 = (Gp @ d - 4) ** 2 + (Gp * (1 - Gp)).sum(1)
check("primer: d0 = 8 -> minimum 0 at y = (1, 1)", v8.min() < 1e-12 and np.allclose(Gp[v8.argmin()], [1, 1]))
# on y1 = 1 the objective is 24 y2^2 - 9 y2 + 1, minimised at y2 = 3/16 with value 5/32
check("primer: d0 = 4 -> minimum 5/32 at y = (1, 3/16)", abs(v4.min() - 5 / 32) < 1e-7 and np.allclose(Gp[v4.argmin()], [1, 3 / 16]), f"{v4.min():.6f} at {Gp[v4.argmin()]}")
check("primer: 5/32 = F(1, 3/16) exactly", abs(F(np.array([1, 3 / 16]), 4) - 5 / 32) < 1e-15)
Qc = np.array([[1, 2], [2, 1]])
check("primer: [[1,2],[2,1]] has eigenvalues -1 and 3", np.allclose(np.linalg.eigvalsh(Qc), [-1, 3]))
check("primer: ... yet x^T Q x >= 0 on the orthant (copositive)", np.einsum("ni,ij,nj->n", simplex, Qc, simplex).min() >= 0)
done()
