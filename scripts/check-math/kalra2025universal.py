"""Kalra et al. 2025: UV model; reduced (f, lambda) map; eta_c; bifurcation; n=1 bouncing valley."""
from common import *

print("kalra2025universal")


def uv_full(n, p, d, y, eta, steps, sw=1.0, seed=0):
    rng = np.random.default_rng(seed)
    ne = n ** (1 - p)
    U = rng.normal(0, sw / np.sqrt(n ** p), (n, d))
    v = rng.normal(0, sw / np.sqrt(n ** p), n)
    x = np.ones(d) / np.sqrt(d)
    out = []
    for _ in range(steps):
        Ux = U @ x
        fx = v @ Ux / np.sqrt(ne)
        r = fx - y
        lam = (Ux @ Ux + (x @ x) * (v @ v)) / ne
        out.append((r, lam))
        U, v = U - eta * r * np.outer(v, x) / np.sqrt(ne), v - eta * r * Ux / np.sqrt(ne)
    return np.array(out), ne, x @ x


def uv_map(r, lam, eta, ne, x2, y):
    r1 = r * (1 - eta * lam + eta ** 2 * x2 / ne * r * (r + y))
    l1 = lam + eta * x2 * r / ne * (eta * lam * r - 4 * (r + y))
    return r1, l1


for (n, p, eta) in [(512, 1.0, .2), (64, .5, .1), (8, 0.0, .05)]:
    traj, ne, x2 = uv_full(n, p, 3, 2.0, eta, 60)
    rr, ll = traj[0]
    err = 0
    for k in range(1, 60):
        rr, ll = uv_map(rr, ll, eta, ne, x2, 2.0)
        err = max(err, abs(rr - traj[k, 0]) + abs(ll - traj[k, 1]))
        rr, ll = traj[k]  # one-step check from the true state
    check(f"Eqs.(1)-(2) reproduce the full UV model (n={n}, p={p})", err < 1e-9, f"max err {err:.1e}")
# Hessian trace = lambda for the scalar n=1 model L(u,v)=1/2(uv-y)^2 at a zero-loss point
y = 2.0
L = lambda q: .5 * (q[0] * q[1] - y) ** 2
Lg = lambda q: np.array([(q[0] * q[1] - y) * q[1], (q[0] * q[1] - y) * q[0]])
check_grad("n=1 UV loss", L, Lg, rand_pts(30, -2.5, 2.5))
q = np.array([np.sqrt(2), np.sqrt(2)])
ev = eig(fd_hess(L, q))
check("on the valley: sharpness = u^2+v^2 (other eigenvalue 0)", abs(ev[1] - 4) < 1e-3 and abs(ev[0]) < 1e-3, str(np.round(ev, 4)))
eta_c = 1 / (1 * y)
print(f"  eta_c = sqrt(n_eff)/(x y) = {eta_c}")
check("flattest valley point has sharpness 2*y = 4 = 2/eta_c", abs(2 * y - 2 / eta_c) < 1e-12)


def late_lams(eta, steps=6000, keep=64, init=(.3, 2.4)):
    qq = np.array(init, float)
    lams = []
    for k in range(steps):
        qq = qq - eta * Lg(qq)
        if not np.all(np.isfinite(qq)) or np.max(np.abs(qq)) > 1e3:
            return None
        if k >= steps - keep:
            lams.append(qq @ qq)
    return np.array(lams)


for eta in [.3, .45, .55, .62, .7, .75]:
    l = late_lams(eta)
    if l is None:
        print(f"  eta={eta}: diverged")
        continue
    uniq = len(np.unique(np.round(l, 4)))
    print(f"  eta={eta}: late sharpness in [{l.min():.3f},{l.max():.3f}], {uniq} distinct values, 2/eta={2 / eta:.3f}")
l = late_lams(.45)
check("eta < eta_c: converges to a valley point with sharpness < 2/eta", np.ptp(l) < 1e-6 and l[-1] < 2 / .45)
l = late_lams(.55)
check("eta > eta_c: keeps oscillating (edge of stability)", np.ptp(l) > 1e-3)
# bifurcation diagram from Eqs. (1)-(2), n_eff=1, x=1, y=2
diverge_at = None
for eta in np.linspace(.3, 1.0, 141):
    r, lam = -1.9, 1.0
    ok = True
    for k in range(4000):
        r, lam = uv_map(r, lam, eta, 1.0, 1.0, 2.0)
        if not np.isfinite(r) or abs(r) > 1e6:
            ok = False
            break
    if not ok:
        diverge_at = eta
        break
print(f"  bifurcation: first divergence at eta ~ {diverge_at:.3f}")
check("diagram extends to ~0.8 before divergence (paper Fig. 3b)", diverge_at is not None and .7 < diverge_at < .95)
# in-page notes (papers/kalra2025universal/notes.md): Sec. 4 worked example (y = 1, n_eff = 1, x = 1)
def uv1(u, v, eta, yy=1.):
    r = u * v - yy
    return u - eta * r * v, v - eta * r * u
u1, v1 = uv1(.1, .1, .2)
check("notes: u = v = 0.1, eta = 0.2: residual -0.99, u+ = v+ = 0.1198, lambda 0.0200 -> 0.02870",
      np.isclose(.1 * .1 - 1, -.99) and np.isclose(u1, .1198) and np.isclose(v1, .1198) and np.isclose(.1 ** 2 + .1 ** 2, .02) and np.isclose(u1 ** 2 + v1 ** 2, .02870, atol=5e-6))
u2, v2 = uv1(2., 2., .1)
check("notes: u = v = 2, eta = 0.1: r = 3, u+ = v+ = 1.4, lambda 8 -> 3.92", np.isclose(2 * 2 - 1, 3) and np.isclose(u2, 1.4) and np.isclose(u2 ** 2 + v2 ** 2, 3.92))
# Sec. 5 Eqs. (1)-(2) reproduce a full UV-model GD step (random n = 7, d = 3, p = 0.5)
Rk = np.random.default_rng(3); nn, dd, pp, yy, et = 7, 3, .5, 2., .05
ne = nn ** (1 - pp); Uk, vk, xk = Rk.normal(size=(nn, dd)) / np.sqrt(nn), Rk.normal(size=nn) / np.sqrt(nn), Rk.normal(size=dd)
fk = lambda U, v: v @ U @ xk / np.sqrt(ne); lamk = lambda U, v: (np.sum((U @ xk) ** 2) + v @ v * (xk @ xk)) / ne
for _ in range(5):
    r0, l0 = fk(Uk, vk) - yy, lamk(Uk, vk)
    Uk, vk = Uk - et * r0 * np.outer(vk, xk) / np.sqrt(ne), vk - et * r0 * (Uk @ xk) / np.sqrt(ne)
    r1, l1 = fk(Uk, vk) - yy, lamk(Uk, vk); x2 = xk @ xk
    assert np.isclose(r1, r0 * (1 - et * l0 + et ** 2 * x2 / ne * r0 * (r0 + yy)), atol=1e-12), (r1,)
    assert np.isclose(l1, l0 + et * x2 / ne * r0 ** 2 * (et * l0 - 4 * (r0 + yy) / r0), atol=1e-12), (l1,)
check("notes: Eqs. (1)-(2) in (r, lambda) match the parameter-space GD step exactly", True)
check("notes: (r, lambda) = (-y, 0) is a fixed point (all parameters zero)", np.isclose(-yy * (1 - 0 + 0), -yy) and np.isclose(0 + 0, 0))
check("notes: eta_c = sqrt(16)/(1 * 2) = 2; eta = 1 below, eta = 2.5 above", np.isclose(np.sqrt(16) / 2, 2) and 1 < 2 < 2.5)
# Sec. 8 cubic analogue: multiplier 1 - 2 eta at z* = 1, = -1 at eta = 1
gz = lambda z, e: z + e * z * (1 - z * z)
check("notes: z_{t+1} = z_t - eta F'(z_t) for F = (z^2 - 1)^2 / 4", np.isclose(gz(.7, .3), .7 - .3 * fd_grad(lambda q: .25 * (q[0] ** 2 - 1) ** 2, np.array([.7]))[0]))
check("notes: multiplier at z* = 1 is 1 - 2 eta (-1 at eta = 1)", all(np.isclose((gz(1 + 1e-7, e) - gz(1 - 1e-7, e)) / 2e-7, 1 - 2 * e, atol=1e-6) for e in [.3, 1.]))
# primer: first-order change -eta lambda r = 0.00396, second-order eta^2 r^2 (r + y) = 0.000392, new output 0.014352
r0 = -.99
check("primer: output change 0.004352 = 0.00396 (first order) + 0.000392 (Eq. 1 second-order term)",
      np.isclose(u1 * v1 - .01, .004352, atol=5e-7) and np.isclose(-.2 * .02 * r0, .00396) and np.isclose(.2 ** 2 * r0 ** 2 * (r0 + 1), .000392, atol=5e-7) and np.isclose(u1 * v1, .014352, atol=5e-7))
def orbit(e, n0=20000, k=64):
    z = .5
    for _ in range(n0): z = gz(z, e)
    o = []
    for _ in range(k): z = gz(z, e); o.append(z)
    return np.array(o)
o09, o11, o125, o135 = orbit(.9), orbit(1.1), orbit(1.25), orbit(1.35)
n_distinct = lambda o: len(np.unique(np.round(o, 6)))
check("primer: eta = 0.9 -> 1; 1.1 -> 2-cycle 0.815 / 1.116; 1.25 -> 4-cycle; 1.35 -> no cycle of length <= 32",
      np.allclose(o09, 1) and n_distinct(o11) == 2 and np.allclose(sorted(np.unique(np.round(o11, 3))), [.815, 1.116]) and n_distinct(o125) == 4
      and all(not np.allclose(o135[k:], o135[:-k], atol=1e-6) for k in range(1, 33)))
check("primer: n = 512, ||x|| = 1, y = 2: eta_c ~ 11.3 (p = 0), 0.5 (p = 1)", np.isclose(np.sqrt(512) / 2, 11.3, atol=.05) and np.isclose(np.sqrt(512 ** 0) / 2, .5))
done()
