"""Mertikopoulos, Hallak, Kavis, Cevher 2020 (NeurIPS; arXiv 2006.11144).
Theorem 2: gamma_n = Theta(1/n^p), p in (2/(q+2), 1] => X_n converges a.s. to a connected component of crit f (Assumptions 1-4).
Theorem 3: gamma_n ~ 1/n^p, p in (0, 1], bounded uniformly exciting noise (Assumption 5, q = inf) => P(X_n -> S) = 0 for every
strict saddle MANIFOLD S (non-isolated saddles allowed; Fig. 1: a ridge).  Theorem 4: near a regular minimizer E|X_n - x*|^2 = O(1/n^p).

Toy: radial f = phi(r), phi(r) = r^4/4 - r^3 + r^2, phi'(r) = r (r - 1)(r - 2): minimum at 0, a RIDGE of strict saddles on the
circle r = 1 (Hessian eigenvalues -1 radial, 0 tangential), a ring of minima at r = 2. Noise: sigma * (uniform unit vector).
"""
from common import *

print("mertikopoulos2020almost")
phi = lambda r_: r_ ** 4 / 4 - r_ ** 3 + r_ ** 2
f = lambda p: phi(np.hypot(p[0], p[1]))
def g(p):
    rr = np.hypot(p[0], p[1]); dphi = rr ** 3 - 3 * rr ** 2 + 2 * rr
    return np.zeros(2) if rr == 0 else dphi / rr * np.asarray(p, float)
check_grad("radial f", f, g, rand_pts(20, -2.5, 2.5))
check_hess_at("at 0 = 2I (regular minimizer)", f, [0, 0], [[2, 0], [0, 2]], tol=2e-3)
th = .7; Hr = fd_hess(f, [np.cos(th), np.sin(th)]); ev = eig(Hr)
check("on the ridge r = 1: gradient 0, Hessian eigenvalues (-1, 0): a strict saddle manifold", np.linalg.norm(g([np.cos(th), np.sin(th)])) < 1e-12 and abs(ev[0] + 1) < 1e-3 and abs(ev[1]) < 1e-3, f"{np.round(ev, 4)}")
H2 = eig(fd_hess(f, [2 * np.cos(th), 2 * np.sin(th)]))
check("ring r = 2: minima (eigenvalues 2 radial, 0 tangential), f = 0", abs(H2[1] - 2) < 1e-3 and abs(H2[0]) < 1e-3 and abs(phi(2.)) < 1e-12)
SIG, GAM, M = .5, 1., 10
def sgd(X, p, n, seed, gd=False):
    R = np.random.default_rng(seed); X = np.array(X, float)
    for k in range(n):
        gam = GAM / (k + M) ** p * M ** p
        rr = np.linalg.norm(X, axis=1, keepdims=True); G = (rr ** 2 - 3 * rr + 2) * X
        if not gd:
            a = R.uniform(0, 2 * np.pi, len(X)); G = G + SIG * np.stack([np.cos(a), np.sin(a)], 1)
        X = X - gam * G
    return X
ang = np.linspace(0, 2 * np.pi, 200, endpoint=False); X0 = np.stack([np.cos(ang), np.sin(ang)], 1)
Xg = sgd(np.array([[1., 0], [0, 1.], [-1., 0], [0, -1.]]), .75, 3000, 0, gd=True)
check("GD started exactly on the ridge (at (+-1, 0), (0, +-1)) stays there: the ridge is its own stable set", np.max(np.abs(np.linalg.norm(Xg, axis=1) - 1)) == 0)
res = {}
for p_ in [.5, .75, 1.]:
    Xe = sgd(X0, p_, 20000, 1)
    rr = np.linalg.norm(Xe, axis=1); inner = np.mean(rr < .2); outer = np.mean(np.abs(rr - 2) < .2); ridge = np.mean(np.abs(rr - 1) < .2)
    res[p_] = (float(inner), float(outer), float(ridge))
    print(f"  SGD from 200 points on the ridge, gamma_n ~ 1/n^{p_}, 20000 steps: near the origin {inner:.3f}, near the outer ring {outer:.3f}, near the ridge {ridge:.3f}")
check("SGD leaves the strict saddle manifold for every schedule (Theorem 3, p in (0, 1])", all(v[2] == 0 for v in res.values()) and all(v[0] + v[1] == 1 for v in res.values()))
# Theorem 4: local rate E|X_n - 0|^2 = O(1/n^p) near the regular minimizer 0, runs started at 0
rates = {}; SL, GL = .1, .5        # smaller noise and step for the local experiment (Theorem 4 is local, conditioned on staying in U)
ns = np.unique(np.round(np.logspace(1, 5, 41)).astype(int))
for p_ in [.5, .75, 1.]:
    R = np.random.default_rng(5); X = np.zeros((2000, 2)); out = []; k = 0; stay = np.ones(len(X), bool)
    for n in ns:
        while k < n:
            gam = GL / (k + M) ** p_ * M ** p_; rr = np.linalg.norm(X, axis=1, keepdims=True)
            a = R.uniform(0, 2 * np.pi, len(X)); X = X - gam * ((rr ** 2 - 3 * rr + 2) * X + SL * np.stack([np.cos(a), np.sin(a)], 1)); k += 1
            stay &= np.linalg.norm(X, axis=1) < .5
        out.append([int(n), float(np.mean(np.sum(X[stay] ** 2, 1)))])
    print(f"  Theorem 4, p = {p_}: runs that stayed in U = {{|x| < 0.5}}: {stay.mean():.3f}")
    out = np.array(out); sel = out[:, 0] >= 1000
    slope = np.polyfit(np.log(out[sel, 0]), np.log(out[sel, 1]), 1)[0]
    rates[str(p_)] = [[int(a_), float(f"{b_:.4g}")] for a_, b_ in out]
    print(f"  Theorem 4, p = {p_}: E|X_n|^2 at n = 1e5: {out[-1, 1]:.3g}, fitted slope over n in [1e3, 1e5]: {slope:.3f} (theory -p = {-p_})")
    check(f"local rate slope ~ -p for p = {p_}", abs(slope + p_) < .05)
save_json("mertikopoulos2020almost", "data.json", {"res": {str(k_): v for k_, v in res.items()}, "rates": rates, "sigma": SIG, "local": {"sigma": SL, "gamma": GL}})
done()
