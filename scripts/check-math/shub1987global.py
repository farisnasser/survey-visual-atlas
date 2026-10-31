"""Shub 1987 (Global Stability of Dynamical Systems, Springer): stable manifold theorem for a hyperbolic fixed point (Ch. 5) and the
center-stable manifold theorem (Appendix III, Thm III.7, the version Lee et al. 2016 use): near a fixed point with an expanding
direction, the points that converge to it lie on a manifold tangent to the contracting (center-stable) eigenspace, of lower
dimension, hence of measure zero.
Toy: GD map G(p) = p - eta grad f(p), eta = 0.1, on f = x^2/2 - u^2/2 + u^4/4 with u = y - k x^2 (k = 0.8): saddle at 0 with
DG(0) = diag(0.9, 1.1); minima (0, +-1). The stable manifold is a CURVE through 0 tangent to the x-axis; we compute it by bisection.
"""
from common import *

print("shub1987global")
K, ETA = .8, .1
def f(p): x, y = p[..., 0], p[..., 1]; u = y - K * x * x; return x * x / 2 - u * u / 2 + u ** 4 / 4
def g(p):
    p = np.asarray(p, float); x, y = p[..., 0], p[..., 1]; u = y - K * x * x; w = -u + u ** 3
    return np.stack([x - 2 * K * x * w, w], -1)
check_grad("f", lambda p: float(f(np.asarray(p))), g, rand_pts(20, -1.5, 1.5))
H0 = check_hess_at("at 0 = diag(1, -1)", lambda p: float(f(np.asarray(p))), [0, 0], [[1, 0], [0, -1]])
DG = np.eye(2) - ETA * H0
check("DG(0) = I - eta H has eigenvalues 0.9 (contracting) and 1.1 (expanding): hyperbolic fixed point", np.allclose(np.sort(eig(DG)), [.9, 1.1], atol=1e-4))
check("minima (0, +-1), f = -1/4", all(np.linalg.norm(g([0, s])) < 1e-12 and abs(f(np.array([0., s])) + .25) < 1e-12 for s in (1., -1.)))
G = lambda P: P - ETA * g(P)
def side(P, T=400):
    for _ in range(T): P = G(P)
    return np.sign(P[..., 1] - K * P[..., 0] ** 2)
xs = np.linspace(-1.2, 1.2, 49)
lo = K * xs ** 2 - .9; hi = K * xs ** 2 + .9          # u = y - k x^2 in [-0.9, 0.9] at the ends
slo = side(np.stack([xs, lo], 1))
check("the two ends of every bisection segment go to different minima", np.all(slo != side(np.stack([xs, hi], 1))))
for _ in range(55):
    mid = (lo + hi) / 2; sm = side(np.stack([xs, mid], 1)); same = sm == slo
    lo = np.where(same, mid, lo); hi = np.where(same, hi, mid)
ystar = (lo + hi) / 2
a = np.polyfit(xs[np.abs(xs) < .3], ystar[np.abs(xs) < .3], 2)
print(f"  stable manifold y*(x): y*(0) = {ystar[24]:.2e}, quadratic fit near 0: {a[0]:.4f} x^2 + {a[1]:.2e} x + {a[2]:.2e}; y*(+-1.2) = {ystar[0]:.4f}, {ystar[-1]:.4f}")
sm_ = (np.abs(xs) <= .25) & (xs != 0)
print(f"  max |y*(x)| / x^2 for 0 < |x| <= 0.25: {np.max(np.abs(ystar[sm_]) / xs[sm_] ** 2):.4f}")
check("it passes through the saddle and is tangent to the contracting direction (the x-axis): |y*(x)| <= 0.35 x^2 near 0", abs(ystar[24]) < 1e-9 and np.all(np.abs(ystar[sm_]) <= .35 * xs[sm_] ** 2))
check("it is curved (not the eigen-line): |y*(1)| > 0.3", abs(np.interp(1., xs, ystar)) > .3)
# starting on it converges to the saddle (for a while, until rounding pushes it off); 1e-6 off it escapes, with time ~ log(1/delta)
P = np.array([xs[40], ystar[40]]); dmin = np.inf
for t in range(300): P = G(P); dmin = min(dmin, np.linalg.norm(P))
print(f"  start on the computed manifold at x = {xs[40]:.2f}: closest approach to the saddle {dmin:.2e}")
check("a start on the computed manifold comes within 1e-6 of the saddle", dmin < 1e-6)
esc = []
for dl in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
    P = np.array([xs[40], ystar[40] + dl]); t = 0
    while abs(P[1] - K * P[0] ** 2) < .5 and t < 5000: P = G(P); t += 1
    esc.append([dl, t])
print(f"  escape steps from delta above the manifold: {esc}")
check("escape time grows like log(1/delta) / log(1.1): about 24 more steps per factor 100", all(15 < esc[i + 1][1] - esc[i][1] < 70 for i in range(4)))
# in-page notes (papers/shub1987global/notes.md): notes Sec. 3 example and primer numbers
p_ = np.array([.7, .3])
for _ in range(50):
    p_ = np.array([.8 * p_[0], 1.2 * p_[1]])
check("notes: GD on (x^2 - y^2)/2 with eta = 0.2 is (0.8x, 1.2y)", np.allclose(np.array([.7, .3]) - .2 * np.array([.7, -.3]), [.8 * .7, 1.2 * .3]))
check("notes: (0.8^k x0, 1.2^k y0): y0 != 0 escapes", abs(p_[0] - .8 ** 50 * .7) < 1e-12 and abs(p_[1]) > 1e3)
A_ = np.array([[.8, .3], [0, 1.2]])
check("primer 1: eigenvalues 0.8, 1.2 with eigenvectors (1, 0) and (0.6, 0.8)", np.allclose(A_ @ [1, 0], [.8, 0]) and np.allclose(A_ @ [.6, .8], 1.2 * np.array([.6, .8])))
check("primer 1: start on the x-axis converges, start with a (0.6, 0.8) component does not", np.linalg.norm(np.linalg.matrix_power(A_, 80) @ [1, 0]) < 1e-6 and np.linalg.norm(np.linalg.matrix_power(A_, 80) @ [1, 1e-3]) > 1)
gq_ = lambda q: np.array([.8 * q[0], 1.2 * q[1] - .28 * q[0] ** 2])
q_ = np.array([1., .5]); its = []
for _ in range(3):
    q_ = gq_(q_); its.append(q_.copy())
check("primer 2: (1, 0.5) -> (0.8, 0.32) -> (0.64, 0.2048) -> (0.512, 0.131), all on y = 0.5 x^2", np.allclose(its, [[.8, .32], [.64, .2048], [.512, .131072]]) and all(abs(t[1] - .5 * t[0] ** 2) < 1e-12 for t in its))
for q0 in rand_pts(5, -1, 1, seed=4):
    u0 = q0[1] - .5 * q0[0] ** 2; q1 = gq_(q0)
    assert abs((q1[1] - .5 * q1[0] ** 2) - 1.2 * u0) < 1e-12
check("primer 2: distance from the parabola, u = y - 0.5x^2, is multiplied by 1.2", True)
check("primer 2: Dg(0) = diag(0.8, 1.2)", np.allclose(np.column_stack([(gq_(e * 1e-6) - gq_(-e * 1e-6)) / 2e-6 for e in np.eye(2)]), np.diag([.8, 1.2])))
x_ = .5
for _ in range(100):
    x_ = x_ - x_ ** 3
check("primer 4: x - x^3 from 0.5: after 100 steps ~0.069, close to 1/sqrt(2k)", abs(x_ - .069) < 1e-3 and abs(x_ - 1 / np.sqrt(200)) < .003, f"{x_:.4f} vs {1 / np.sqrt(200):.4f}")
ok4 = True
for x0 in np.linspace(-.99, .99, 199):
    x = x0
    for _ in range(20000):
        x = x - x ** 3
    ok4 &= abs(x) < .01
check("primer 4: every start on y = 0 with |x| < 1 converges to 0", ok4)
save_json("shub1987global", "data.json", {"xs": r(xs, 4), "ys": [float(v) for v in ystar], "esc": esc, "fit": a[0]})
done()
