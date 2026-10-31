"""Crash course stop 4 (course/curvature): second derivative, Hessian, eigen-directions, GD per eigen-direction,
condition number, strong convexity rate, Hessian-vector products."""
from common import *

print("course_curvature")
# ---- 1. second derivative of the stop-1 loss f = 0.1x^2 + 0.5 sin 3x (visual 'bend')
F = lambda x: .1 * x * x + .5 * np.sin(3 * x)
dF = lambda x: .2 * x + 1.5 * np.cos(3 * x)
d2F = lambda x: .2 - 4.5 * np.sin(3 * x)
xs = np.linspace(-3, 3, 600001)
check("f'' = derivative of f' (finite differences)", np.allclose((dF(xs[1:-1] + 1e-6) - dF(xs[1:-1] - 1e-6)) / 2e-6, d2F(xs[1:-1]), atol=1e-6))
check("text: f''(-0.5) = 4.689 (bends up), f''(0.5) = -4.289 (bends down)", round(d2F(-.5), 3) == 4.689 and round(d2F(.5), 3) == -4.289)
check("text: f'' ranges over [-4.3, 4.7] on the real line, so L = 4.7", np.isclose(d2F(xs).min(), -4.3, atol=1e-6) and np.isclose(d2F(xs).max(), 4.7, atol=1e-6))
# 1-D quadratic model error vs tangent error at x = -0.5, step 0.3
x0, h = -.5, .3
lin = F(x0) + dF(x0) * h; quad = lin + .5 * d2F(x0) * h * h
e1, e2 = abs(F(x0 + h) - lin), abs(F(x0 + h) - quad)
print(f"  at x = -0.5, step 0.3: tangent error {e1:.4f}, parabola error {e2:.4f}")
check("text: tangent error 0.194, parabola error 0.017 at x = -0.5, h = 0.3", round(e1, 3) == .194 and round(e2, 3) == .017)

# ---- 2. Hessian example f = 2x^2 + 2xy + 2y^2
fq = lambda p: 2 * p[0] ** 2 + 2 * p[0] * p[1] + 2 * p[1] ** 2
gq = lambda p: np.array([4 * p[0] + 2 * p[1], 2 * p[0] + 4 * p[1]])
check_grad("2x^2 + 2xy + 2y^2", fq, gq, rand_pts(10, -2, 2))
Hq = check_hess_at("2x^2 + 2xy + 2y^2", fq, [.3, -.7], [[4, 2], [2, 4]])
lam, V = np.linalg.eigh(np.array([[4., 2], [2, 4]]))
check("eigenvalues 2 and 6, eigenvectors (1,-1)/sqrt2 and (1,1)/sqrt2", np.allclose(lam, [2, 6]) and np.isclose(abs(V[:, 0] @ [1, -1]) / np.sqrt(2), 1) and np.isclose(abs(V[:, 1] @ [1, 1]) / np.sqrt(2), 1))
H = np.array([[4., 2], [2, 4]])
u = np.array([1., 0])
check("curvature along x: u^T H u = 4 = 6 cos^2 45 + 2 sin^2 45", np.isclose(u @ H @ u, 4) and np.isclose(6 * .5 + 2 * .5, 4))
# second directional derivative = u^T H u (finite differences along a line)
for ang in np.linspace(0, np.pi, 7):
    u = np.array([np.cos(ang), np.sin(ang)]); p = np.array([.4, -.2]); t = 1e-3
    sd = (fq(p + t * u) - 2 * fq(p) + fq(p - t * u)) / t ** 2
    assert np.isclose(sd, u @ H @ u, atol=1e-5)
check("d^2/dt^2 f(x + t u) = u^T H u", True)
# quadratic form in eigen coordinates: u^T H u = sum lam_i (u.v_i)^2 lies in [lam_min, lam_max]
R = np.random.default_rng(1)
ok = True
for _ in range(200):
    A = R.normal(size=(4, 4)); S = (A + A.T) / 2; w, Q = np.linalg.eigh(S); u = R.normal(size=4); u /= np.linalg.norm(u)
    ok &= np.isclose(u @ S @ u, np.sum(w * (Q.T @ u) ** 2)) and w[0] - 1e-12 <= u @ S @ u <= w[-1] + 1e-12
check("u^T H u = sum lam_i (u.v_i)^2, always between lam_min and lam_max", ok)
# visual 'quad': H = R diag(l1, l2) R^T; default l1 = 6, l2 = 2, theta = 45 deg gives [[4,2],[2,4]]
def Hrot(l1, l2, th):
    c, s = np.cos(th), np.sin(th); Rm = np.array([[c, -s], [s, c]])
    return Rm @ np.diag([l1, l2]) @ Rm.T
check("visual 'quad' default (6, 2, 45 deg) is the worked example", np.allclose(Hrot(6, 2, np.pi / 4), H))
check("quad display: max |height| on the box |x|,|y| <= 1.3 is 0.1 * 10.14 = 1.01 world units", np.isclose(.5 * 6 * 2 * 1.3 ** 2, 10.14))
check("bend readout: f'(-0.5) = 0.006", round(dF(-.5), 3) == .006)

# ---- 4. GD on a quadratic decouples along eigen-directions (visual 'zigzag')
TH = np.radians(30)
v1 = np.array([np.cos(TH), np.sin(TH)]); v2 = np.array([-np.sin(TH), np.cos(TH)])
def zz(kappa, eta, x0=(-2.2, -.6), T=400):
    Hz = 1 * np.outer(v1, v1) + kappa * np.outer(v2, v2)
    fz = lambda p: .5 * p @ Hz @ p
    x = np.array(x0, float); u10, u20 = v1 @ x, v2 @ x; hit = None
    for t in range(1, T + 1):
        x = x - eta * Hz @ x
        assert np.isclose(v1 @ x, (1 - eta) ** t * u10, atol=1e-9) and np.isclose(v2 @ x, (1 - eta * kappa) ** t * u20, atol=1e-6 * max(1, abs(1 - eta * kappa) ** t))
        if hit is None and fz(x) < 1e-6:
            hit = t
        if np.max(np.abs(x)) > 50:
            return hit, True
    return hit, False
hit, div = zz(10, .18)
print(f"  zigzag default kappa 10, eta 0.18: f < 1e-6 after {hit} steps")
check("u_i <- (1 - eta lam_i) u_i exactly, for kappa 10, eta 0.18 (default): factors 0.82 and -0.8", hit is not None and not div and np.isclose(1 - .18, .82) and np.isclose(1 - 1.8, -.8))
check("zigzag default: f < 1e-6 after 38 steps", hit == 38)
check("zigzag: eta = 0.21 > 2/L = 0.2 diverges", zz(10, .21)[1])
for k, e in [(4, .1), (25, .05), (10, .1)]:
    zz(k, e)
check("decoupling holds for other kappa, eta", True)
# u1 and u2 of the default start (readout uses them)
x0 = np.array([-2.2, -.6])
print(f"  start in eigen coordinates: u1 = {v1 @ x0:.3f}, u2 = {v2 @ x0:.3f}")

# ---- 5. condition number: steps for every eigen-coordinate to shrink by 1e6 (visual 'kappa')
def steps_needed(kappa, eta, tol=1e-6, T=200000):
    u = np.array([1., 1.]); lam = np.array([1., kappa])
    for t in range(1, T + 1):
        u = (1 - eta * lam) * u
        if np.all(np.abs(u) <= tol):
            return t
    return None
pred = lambda q: int(np.ceil(np.log(1e6) / -np.log(q)))
for kappa in [10, 100, 1000]:
    a, b = steps_needed(kappa, 1 / kappa), steps_needed(kappa, 2 / (kappa + 1))
    print(f"  kappa {kappa}: eta = 1/L -> {a} steps, eta = 2/(L+mu) -> {b} steps, kappa ln(1e6) = {kappa * np.log(1e6):.0f}")
check("kappa 10: 132 steps with eta = 1/L (factor 0.9), 69 with eta = 2/(L+mu) (factor 9/11)", steps_needed(10, .1) == 132 == pred(.9) and steps_needed(10, 2 / 11) == 69 == pred(9 / 11))
check("kappa 100: 1375 steps with eta = 1/L", steps_needed(100, .01) == 1375)
check("kappa 1000: 13809 steps with eta = 1/L", steps_needed(1000, .001) == 13809)
check("kappa 1: one step lands on the minimum", steps_needed(1, 1.) == 1)
check("best fixed step 2/(L+mu) gives factor (kappa-1)/(kappa+1)", all(np.isclose(max(abs(1 - 2 / (k + 1)), abs(1 - 2 * k / (k + 1))), (k - 1) / (k + 1)) for k in [2, 10, 100]))
check("steps ~ kappa ln(1/eps): ratio -> 1 as kappa grows", abs(steps_needed(1000, .001) / (1000 * np.log(1e6)) - 1) < .001)

# ---- 6. strong convexity: PL from strong convexity, linear rate (1 - mu/L)^t, on a real non-quadratic example
# f = x^2/2 + y^2 + 0.2 sin(x) sin(y)... use f = 0.5 x^2 + 2 y^2 + 0.1 cos(x + y): f'' between mu and L
fs = lambda p: .5 * p[0] ** 2 + 2 * p[1] ** 2 + .1 * np.cos(p[0] + p[1])
gs = lambda p: np.array([p[0] - .1 * np.sin(p[0] + p[1]), 4 * p[1] - .1 * np.sin(p[0] + p[1])])
check_grad("strongly convex example", fs, gs, rand_pts(10, -2, 2))
# Hessian = diag(1, 4) - 0.1 cos(x+y) [[1,1],[1,1]] -> eigenvalues within [1 - 0.2, 4 + 0.2]... compute exactly over a grid
ev = np.array([eig(np.diag([1., 4]) - .1 * c * np.ones((2, 2))) for c in np.linspace(-1, 1, 2001)])
mu, Ls = ev[:, 0].min(), ev[:, 1].max()
print(f"  example: mu = {mu:.4f}, L = {Ls:.4f}")
xs_ = gd(gs, [2, -1.5], 1 / Ls, 60)
fst = min(fs(p) for p in [gd(gs, [2, -1.5], 1 / Ls, 3000)[-1]])
gap = np.array([fs(p) - fst for p in xs_])
check("f(x_t) - f* <= (1 - mu/L)^t (f(x0) - f*) along GD with eta = 1/L", np.all(gap <= (1 - mu / Ls) ** np.arange(61) * gap[0] + 1e-12))
check("PL: |grad f|^2 >= 2 mu (f - f*) on random points", all(np.sum(gs(p) ** 2) >= 2 * mu * (fs(p) - fst) - 1e-12 for p in rand_pts(500, -3, 3)))

# ---- 7. Hessian-vector products (visual 'hvp') on stop 2's surface fp
fp = lambda p: .3 * p[0] ** 2 + .2 * p[1] ** 2 + .4 * np.sin(1.5 * p[0]) * np.cos(p[1])
gp = lambda p: np.array([.6 * p[0] + .6 * np.cos(1.5 * p[0]) * np.cos(p[1]), .4 * p[1] - .4 * np.sin(1.5 * p[0]) * np.sin(p[1])])
Hp = lambda p: np.array([[.6 - .9 * np.sin(1.5 * p[0]) * np.cos(p[1]), -.6 * np.cos(1.5 * p[0]) * np.sin(p[1])],
                         [-.6 * np.cos(1.5 * p[0]) * np.sin(p[1]), .4 - .4 * np.sin(1.5 * p[0]) * np.cos(p[1])]])
check_grad("stop 2 surface", fp, gp, rand_pts(10, -2, 2))
P0 = np.array([1.1, .9]); vv = np.array([1, 1]) / np.sqrt(2)
check_hess_at("stop 2 surface at (1.1, 0.9)", fp, P0, Hp(P0))
Hv = Hp(P0) @ vv
print(f"  exact Hv at (1.1, 0.9), v = (1,1)/sqrt2: {np.round(Hv, 4)}")
hs = 10.0 ** np.arange(-12, 0.01, .25)
fwd = [np.linalg.norm((gp(P0 + h * vv) - gp(P0)) / h - Hv) for h in hs]
cen = [np.linalg.norm((gp(P0 + h * vv) - gp(P0 - h * vv)) / (2 * h) - Hv) for h in hs]
hf, hc = hs[int(np.argmin(fwd))], hs[int(np.argmin(cen))]
print(f"  forward difference best h = {hf:.1e} (err {min(fwd):.1e}); central best h = {hc:.1e} (err {min(cen):.1e})")
check("text: forward difference is best near h ~ 1e-8 with error ~ 1e-8", 1e-9 <= hf <= 1e-7 and 1e-10 < min(fwd) < 1e-7)
check("text: central difference is best near h ~ 1e-6 with error ~ 1e-11", 1e-7 <= hc <= 1e-5 and 1e-12 < min(cen) < 1e-10)
fw4 = np.linalg.norm((gp(P0 + 1e-4 * vv) - gp(P0)) / 1e-4 - Hv)
check("forward error at h = 1e-4 is ~1e-5 (shrinks like h)", 1e-6 < fw4 < 1e-4, f"{fw4:.1e}")
# memory of a full Hessian
check("text: d = 1e8 -> d^2 = 1e16 entries, 4 bytes each = 4e16 bytes = 40 petabytes", np.isclose((1e8) ** 2 * 4, 4e16))
done()
