"""Ge, Li, Wang, Wang 2019: Stabilized SVRG (Algorithm 3) vs Perturbed SVRG (Algorithm 2) near a saddle-like point.

f = mean f_i, f_i(x, y) = g0 x + x^2 - y^2/2 + y^4/4 + a_i x + b_i y + c_i (x^2 - y^2)/2 + d_i x^3 / 3, zero-mean a, b, c, d.
At x~ = 0: grad f = (g0, 0) small (<= G), Hessian diag(2, -1): lambda_min = -gamma = -1, e1 = (0, 1).
Components have large Hessian-Lipschitz rho' (the d_i x^3 terms cancel in f): the setting where stabilization matters.
Perturbed SVRG: perturb in B(delta), then SVRG.  Stabilized SVRG: also subtract v_shift = grad f(x~) from every estimate,
i.e. run SVRG on f^(x) = f(x) - <grad f(x~), x - x~>.  Section 4.3: Phase 1 stays within O(delta) of x~ (Fig. 1b, Fig. 2).
"""
from common import *

print("ge2019stabilized")
n = 125
b = int(round(n ** (2 / 3)))   # 25
m = n // b                     # 5
R0 = np.random.default_rng(82)
a, bb, c, d = (R0.normal(0, s, n) for s in (.6, .6, .4, 3.0))
a -= a.mean(); bb -= bb.mean(); c -= c.mean(); d -= d.mean()
g0 = .08
f = lambda p: g0 * p[0] + p[0] ** 2 - .5 * p[1] ** 2 + .25 * p[1] ** 4
g = lambda p: np.array([g0 + 2 * p[0], -p[1] + p[1] ** 3])
gi = lambda p, I: np.stack([g0 + 2 * p[0] + a[I] + c[I] * p[0] + d[I] * p[0] ** 2, -p[1] + p[1] ** 3 + bb[I] - c[I] * p[1]], -1)
fi = lambda p, i: f(p) + a[i] * p[0] + bb[i] * p[1] + c[i] * (p[0] ** 2 - p[1] ** 2) / 2 + d[i] * p[0] ** 3 / 3
check_grad("f", f, g, rand_pts(20, -1.5, 1.5))
check_grad("f_4", lambda p: fi(p, 4), lambda p: gi(p, np.array([4]))[0], rand_pts(20, -1.5, 1.5))
check("mean component gradient = grad f", np.allclose(gi(np.array([.3, .2]), np.arange(n)).mean(0), g([.3, .2])))
check_hess_at("f at x~ = 0", f, np.zeros(2), np.diag([2, -1]))
rho = 6 * 1.5                      # f: d/dy (3y^2 - 1) on |y| <= 1.5
rhop = float(rho + 2 * np.max(np.abs(d)))   # components add d/dx (2 d_i x)
print(f"  rho = {rho:.1f}, rho' = {rhop:.1f}, b = {b}, m = {m}, |grad f(0)| = {g0}")
eta, delta, T, DIST = .1, .02, 120, .3
xs_min = np.array([-g0 / 2, 1.0])   # minimum: 2x + g0 = 0, y^3 - y = 0 -> y = +-1
check("minima at (-g0/2, +-1)", np.allclose(g(xs_min), 0))


def svrg(seed, stabilized, T=T):
    R = np.random.default_rng(seed)
    xt = np.zeros(2)
    shift = g(xt) if stabilized else np.zeros(2)
    th = R.uniform(0, 2 * np.pi); rr = delta * np.sqrt(R.uniform())
    x = xt + rr * np.array([np.cos(th), np.sin(th)])
    path = [x.copy()]
    for s in range(T // m):
        snap = x.copy(); gs = g(snap)
        for t in range(m):
            I = R.integers(0, n, b)
            v = (gi(x, I) - gi(snap, I)).mean(0) + gs - shift
            x = x - eta * v; path.append(x.copy())
            if stabilized and np.linalg.norm(x - xt) > DIST:   # stopping condition: travelled the threshold distance
                shift = np.zeros(2)
    return np.array(path)


P = [svrg(s, False) for s in range(200)]
S_ = [svrg(s, True) for s in range(200)]
K1 = int(1 / (eta * 1))   # 1/(eta gamma) = 10 steps
dP = np.median([np.linalg.norm(p[K1, :] - 0) for p in P]); dS = np.median([np.linalg.norm(p[K1, :] - 0) for p in S_])
check("Phase 1 (t = 1/(eta gamma)): stabilized iterates stay within O(delta) of x~", dS < 3 * delta, f"median |x - x~| = {dS:.4f} (delta = {delta})")
check("unstabilized iterates drift ~ |grad f(x~)|/lambda_+ away along the positive-curvature axis", dP > 1.5 * dS, f"median |x - x~| = {dP:.4f}")
xdrift_P = np.median([p[K1, 0] for p in P]); xdrift_S = np.median([p[K1, 0] for p in S_])
print(f"  x-coordinate at t = {K1}: perturbed {xdrift_P:.4f}, stabilized {xdrift_S:.4f}")
esc = lambda paths: np.mean([abs(p[-1, 1]) > .5 for p in paths])
check("both escape along e1 by t = 120 in most runs", esc(P) > .9 and esc(S_) > .9, f"perturbed {esc(P):.0%}, stabilized {esc(S_):.0%}")
fdec = np.mean([f(p[-1]) - f(np.zeros(2)) for p in S_])
check("stabilized super epoch decreases f (Lemma 7: by ~gamma^3/rho^2 with constant probability)", fdec < -.1, f"mean f(x_T) - f(x~) = {fdec:.3f}")
check("after the super epoch the shift is dropped and the run ends at a minimum of f", np.mean([np.linalg.norm(g(p[-1])) < .05 for p in S_]) > .9)
save_json("ge2019stabilized", "data.json", {"a": r(a, 5), "b": r(bb, 5), "c": r(c, 5), "d": r(d, 5), "g0": g0, "eta": eta, "delta": delta, "dist": DIST, "bs": b, "m": m,
                                           "dP": round(float(dP), 4), "dS": round(float(dS), 4)})
done()
