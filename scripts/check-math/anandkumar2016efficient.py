"""Anandkumar & Ge 2016 (COLT): third-order local minima can be found efficiently (Theorems 8-9), fourth-order ones cannot
(Theorem 10, NP-hard). Definition 1: x is a p-th order local min if f(y) >= f(x) - C|y - x|^{p+1} near x.
Definition 4 (third-order necessary condition): grad = 0, Hessian PSD, and T(u,u,u) = 0 for every u with u^T H u = 0.

third : f3(x, y) = x^2 + y^3 + y^4      origin: H = diag(2, 0) PSD but d^3f/dy^3 = 6  -> 2nd-order min, not 3rd-order
fourth: f4(x, y) = x^2 - y^4 + y^6      origin: passes Definition 4 (T = 0) but f4(0, y) = -y^4 + y^6 < 0 -> 3rd-order, not 4th
"""
from common import *

print("anandkumar2016efficient")
f3 = lambda p: p[0] ** 2 + p[1] ** 3 + p[1] ** 4
g3 = lambda p: np.array([2 * p[0], 3 * p[1] ** 2 + 4 * p[1] ** 3])
f4 = lambda p: p[0] ** 2 - p[1] ** 4 + p[1] ** 6
g4 = lambda p: np.array([2 * p[0], -4 * p[1] ** 3 + 6 * p[1] ** 5])
check_grad("f3 = x^2 + y^3 + y^4", f3, g3, rand_pts(20, -1.2, 1.2))
check_grad("f4 = x^2 - y^4 + y^6", f4, g4, rand_pts(20, -1.2, 1.2))
check_hess_at("f3 at 0 = diag(2, 0)", f3, [0, 0], [[2, 0], [0, 0]])
check_hess_at("f4 at 0 = diag(2, 0)", f4, [0, 0], [[2, 0], [0, 0]])
d3 = lambda f, h=1e-3: (f([0, 2 * h]) - 2 * f([0, h]) + 2 * f([0, -h]) - f([0, -2 * h])) / (2 * h ** 3)   # third derivative along e_y at 0
check("f3: third derivative along the Hessian null direction e_y is 6 != 0 -> Definition 4 fails", abs(d3(f3) - 6) < 1e-4, f"{d3(f3):.5f}")
check("f4: third derivative along e_y is 0 -> Definition 4 holds at 0", abs(d3(f4)) < 1e-6)
ys = np.linspace(1e-3, .05, 50)
check("f3: 2nd-order local min (f >= -C|y|^3) but not 3rd-order (f(0,-t) = -t^3 + t^4 < -C t^4 for small t)", all(f3([0, -t]) >= -2 * t ** 3 for t in ys) and f3([0, -1e-4]) < -100 * 1e-16)
check("f4: 3rd-order local min (f >= -C|y|^4) but not 4th-order (f(0,t) = -t^4 + t^6 < -C t^5 for small t)", all(f4([0, t]) >= -2 * t ** 4 for t in ys) and f4([0, 1e-4]) < -100 * 1e-20)
y3 = -.75; y4 = np.sqrt(2 / 3)
check("f3 local min at (0, -3/4), value -27/256", np.linalg.norm(g3([0, y3])) < 1e-12 and abs(f3([0, y3]) + 27 / 256) < 1e-12 and eig(fd_hess(f3, [0, y3]))[0] > 0)
check("f4 global minima at (0, +-sqrt(2/3)), value -4/27", np.linalg.norm(g4([0, y4])) < 1e-12 and abs(f4([0, y4]) + 4 / 27) < 1e-12 and eig(fd_hess(f4, [0, y4]))[0] > 0)
G = np.linspace(-2, 2, 401); P = np.stack(np.meshgrid(G, G), -1)
check("f4 bounded below on the box by -4/27", (P[..., 0] ** 2 - P[..., 1] ** 4 + P[..., 1] ** 6).min() >= -4 / 27 - 1e-9)
# GD (step .05) on f3 from y0 > 0 creeps into the degenerate saddle: y_k ~ 1/(3 eta k)
lr = .05; p = np.array([.8, .6]); path = gd(g3, p, lr, 4000)
yk = path[-1, 1]
print(f"  f3, GD from (0.8, 0.6): after 4000 steps y = {yk:.5f} (1/(3 eta k) = {1 / (3 * lr * 4000):.5f}), f = {f3(path[-1]):.2e}")
check("f3: GD from y > 0 converges to the non-minimum origin (sublinearly, y_k ~ 1/(3 eta k))", 0 < yk < 2.5 / (3 * lr * 4000) and abs(path[-1, 0]) < 1e-12)
# GD + third-order step (simplified from Sec. 5): when |grad| and the Hessian give no descent, test the cubic form on the
# small-eigenvalue subspace and step against its sign
def gd3(g, f, p, lr, steps, gtol=1e-2, delta=.1):
    p = np.array(p, float); kicks = []
    for k in range(steps):
        gr = g(p)
        if np.linalg.norm(gr) < gtol:
            H = fd_hess(f, p); w, V = np.linalg.eigh(H)
            if w[0] > -1e-3:
                v = V[:, np.argmin(np.abs(w))]; h = 1e-3
                c = (f(p + 2 * h * v) - 2 * f(p + h * v) + 2 * f(p - h * v) - f(p - 2 * h * v)) / (2 * h ** 3)
                if abs(c) > 1:                                  # competitive: cubic term dominates the (tiny) quadratic one
                    p = p - np.sign(c) * delta * v; kicks.append(k); continue
        p = p - lr * gr
    return p, kicks
pe, kicks = gd3(g3, f3, [.8, .6], lr, 4000)
print(f"  f3, GD + third-order step: third-order step at iteration(s) {kicks[:3]}, ends at y = {pe[1]:.4f}, f = {f3(pe):.4f}")
check("f3: the third-order step leaves the degenerate saddle and GD reaches the minimum (0, -3/4)", abs(pe[1] - y3) < 1e-6 and len(kicks) >= 1)
pe4, k4 = gd3(g4, f4, [.8, 1e-9], lr, 4000)
check("f4: the third-order test finds nothing at the 4th-order saddle (no third-order step taken near 0)", len(k4) == 0 and abs(pe4[1]) < 1e-6)
# GD on f4 from y0: escape time ~ 1/(8 eta y0^2) (dy/dk = 4 eta y^3)
ratios = []
for y0 in [.05, .1, .2]:
    path = gd(g4, [.5, y0], lr, 40000); k = int(np.argmax(np.abs(path[:, 1]) > .5)); ratios.append(k * 8 * lr * y0 ** 2)
    print(f"  f4, GD from y0 = {y0}: reaches |y| = 0.5 at step {k} (1/(8 eta y0^2) = {1 / (8 * lr * y0 ** 2):.0f})")
check("f4: GD escape time ~ 1/(8 eta y0^2) (within 3%)", all(abs(q - 1) < .03 for q in ratios))

# in-page notes (papers/anandkumar2016efficient/notes.md): Sec. 2 and Sec. 4 toys and the primer's numbers
f24 = lambda p: p[0] ** 2 - p[1] ** 4
check_hess_at("notes Sec. 2: x^2 - y^4 has Hessian diag(2, 0) at 0", f24, [0, 0], [[2, 0], [0, 0]])
check("notes Sec. 2: f(0, y) = -y^4 < 0", all(f24([0, y]) < 0 for y in (-.3, -.01, .01, .3)))
L3 = lambda p: .5 * (1 - p[0] * p[1] * p[2]) ** 2
check("notes Sec. 4: gradient and Hessian of 1/2(1-abc)^2 vanish at 0", np.allclose(fd_grad(L3, np.zeros(3)), 0, atol=1e-8) and np.allclose(fd_hess(L3, np.zeros(3)), 0, atol=1e-6))
check("notes Sec. 4: L(t,t,t) = 1/2 - t^3 + t^6/2", all(np.isclose(L3([t, t, t]), .5 - t ** 3 + t ** 6 / 2) for t in np.linspace(-1, 1, 21)))
h3 = lambda t: t ** 3  # h(0, y) = y^3
check("primer idea 1: -t^3 >= -C t^3 with C = 1, but -t^3 >= -C t^4 needs C >= 10 at t = 0.1",
      all(-t ** 3 >= -1 * t ** 3 for t in np.linspace(0, .5, 11)) and np.isclose(.1 ** 3 / .1 ** 4, 10))
d3 = lambda g, h=1e-2: (g(2 * h) - 2 * g(h) + 2 * g(-h) - g(-2 * h)) / (2 * h ** 3)
hh = lambda p: p[0] ** 2 + p[1] ** 3
check("primer idea 2: third derivative of h along (0,1) = 6, along (0,-1) = -6",
      np.isclose(d3(lambda t: hh([0, t])), 6) and np.isclose(d3(lambda t: hh([0, -t])), -6))
q1 = lambda u: (u[0] ** 2 - u[1] ** 2) ** 2; q2 = lambda u: u[0] ** 4 + u[1] ** 4 - 3 * u[0] ** 2 * u[1] ** 2
check("primer idea 3: (u1^2-u2^2)^2 >= 0; u1^4+u2^4-3u1^2u2^2 = -1 at (1,1)", min(q1(u) for u in rand_pts(2000, -2, 2)) >= 0 and q2([1, 1]) == -1)
check("primer idea 4: third derivative of L(t,t,t) at 0 is -6", np.isclose(d3(lambda t: L3([t, t, t])), -6, atol=1e-6))
L4 = lambda t: .5 * (1 - t ** 4) ** 2
check("primer idea 4: L(t,t,t,t) = 1/2 - t^4 + t^8/2", all(np.isclose(L4(t), .5 - t ** 4 + t ** 8 / 2) for t in np.linspace(-1, 1, 21)))
done()
