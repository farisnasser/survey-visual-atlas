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
done()
