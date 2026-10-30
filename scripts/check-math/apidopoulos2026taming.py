"""Apidopoulos, Lytras, Mertikopoulos 2026 (ICML, PMLR 306): tamed SGD under (L0, L1)-smoothness.
(TSGD)  x+ = x - gamma_t / (1 + lambda_t r(x)) * g_hat,   r deterministic with r = Omega(max{1, sqrt(f - f*), |grad f|})  (Eq. 12)
Theorem 1: sum gamma = inf, sum gamma^2/lambda < inf, gamma/lambda -> 0  => grad f(x_t) -> 0 a.s.  (gamma ~ t^-p, lambda ~ t^-q: (q+1)/2 < p <= 1)
Theorem 3 + (25a): TSGD converges to crit f and avoids strict saddle manifolds a.s. if 2/3 < p <= 1 and p/2 < q < 2p - 1.

Toy: f(x, y) = x^4/4 - x^2/2 + y^2/2 + y^4/4: strict saddle at 0, minima (+-1, 0), gradient grows like |x|^3 (not globally
Lipschitz), (L0, L1)-smooth and gradient-coercive (Assumptions 1-2). Noise: g_hat = grad f + N(0, s^2 I) (sub-Gaussian, ABC with C = s^2).
"""
from common import *

print("apidopoulos2026taming")
f = lambda p: p[0] ** 4 / 4 - p[0] ** 2 / 2 + p[1] ** 2 / 2 + p[1] ** 4 / 4
g = lambda p: np.array([p[0] ** 3 - p[0], p[1] + p[1] ** 3])
check_grad("f", f, g, rand_pts(20, -3, 3))
check_hess_at("at 0 = diag(-1, 1): strict saddle", f, [0, 0], [[-1, 0], [0, 1]])
check_hess_at("at (1, 0) = diag(2, 1): minimum", f, [1, 0], [[2, 0], [0, 1]])
fstar = -.25
# (L0, L1)-smoothness: |Hess| <= L0 + L1 |grad| on a big box; not globally Lipschitz
P = rand_pts(20000, -30, 30, seed=1)
H = np.maximum(np.abs(3 * P[:, 0] ** 2 - 1), 1 + 3 * P[:, 1] ** 2); G = np.linalg.norm(np.stack([P[:, 0] ** 3 - P[:, 0], P[:, 1] + P[:, 1] ** 3], 1), axis=1)
check("(L0, L1)-smooth with L0 = 4, L1 = 3 on [-30, 30]^2", np.all(H <= 4 + 3 * G))
check("gradient not globally Lipschitz (Hessian norm ~ 3|x|^2 unbounded)", H.max() > 2000)
check("gradient coercive: <grad f, x> / |x|^2 > 0 for |x| >= 2", np.all(((P[:, 0] ** 4 - P[:, 0] ** 2 + P[:, 1] ** 2 + P[:, 1] ** 4) / np.sum(P ** 2, 1))[np.linalg.norm(P, axis=1) >= 2] > 0))
r_ = lambda X: 1 + np.linalg.norm(X, axis=-1) ** 3          # scale function ~ |grad f| at infinity, as in Eq. (12)
p_, q_ = .8, .5
check("schedule p = 0.8, q = 0.5 meets Theorem 1 ((q+1)/2 < p <= 1) and (25a) (2/3 < p <= 1, p/2 < q < 2p - 1)", (q_ + 1) / 2 < p_ <= 1 and 2 / 3 < p_ and p_ / 2 < q_ < 2 * p_ - 1)
SIG, T0 = .3, 10


def run(method, g0, X0, T=5000, seed=0):
    R = np.random.default_rng(seed); X = X0.copy(); alive = np.ones(len(X), bool)
    for t in range(T):
        gam = g0 / ((t + T0) / T0) ** p_; lam = 1 / ((t + T0) / T0) ** q_
        Gr = np.stack([X[:, 0] ** 3 - X[:, 0], X[:, 1] + X[:, 1] ** 3], 1) + SIG * R.normal(size=X.shape)
        if method == "sgd": step = gam * Gr
        elif method == "clip": nG = np.linalg.norm(Gr, axis=1, keepdims=True); step = gam * Gr * np.minimum(1, 10 / np.maximum(nG, 1e-300))
        else: step = gam / (1 + lam * r_(X))[:, None] * Gr
        X = np.where(alive[:, None], X - step, X)
        alive &= np.all(np.isfinite(X), 1) & (np.linalg.norm(X, axis=1) < 1e6)
        X[~alive] = 1e6
    gn = np.linalg.norm(np.stack([X[:, 0] ** 3 - X[:, 0], X[:, 1] + X[:, 1] ** 3], 1), axis=1)
    return X, alive, gn


R0 = np.random.default_rng(7); N = 400
ang = R0.uniform(0, 2 * np.pi, N); rad = R0.uniform(0, 4, N); X0 = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
rows = []
for g0 in [.1, .3, 1., 3., 10.]:
    row = [g0]
    for m in ["sgd", "clip", "tsgd"]:
        X, alive, gn = run(m, g0, X0)
        mins = np.minimum(np.linalg.norm(X - [1, 0], axis=1), np.linalg.norm(X + [1, 0], axis=1))
        row += [float(np.mean(~alive)), float(np.mean(alive & (mins < .15))), float(np.mean(alive & (np.linalg.norm(X, axis=1) < .15)))]
    rows.append(row)
    print(f"  gamma0 = {g0}: diverged / near a minimum / near the saddle  SGD {row[1]:.3f}/{row[2]:.3f}/{row[3]:.3f}  clip {row[4]:.3f}/{row[5]:.3f}/{row[6]:.3f}  TSGD {row[7]:.3f}/{row[8]:.3f}/{row[9]:.3f}")
rows = np.array(rows)
check("plain SGD diverges in a growing fraction of runs once gamma0 >= 0.3", rows[1, 1] > .2 and np.all(np.diff(rows[1:, 1]) >= 0))
check("TSGD never diverges, for every gamma0", np.all(rows[:, 7] == 0))
check("TSGD ends near a minimum (not the saddle) in >= 95% of runs for every gamma0", np.all(rows[:, 8] >= .95) and np.all(rows[:, 9] == 0))
# avoidance from the stable manifold of the saddle (x = 0): noise excites the unstable direction
Xs = np.stack([np.zeros(200), np.linspace(-3, 3, 200)], 1)
X, alive, gn = run("tsgd", .3, Xs, T=6000, seed=3)
near_saddle = np.mean(np.linalg.norm(X, axis=1) < .15)
print(f"  TSGD from 200 starts on the stable manifold x = 0: near the saddle at the end {near_saddle:.3f}, near a minimum {np.mean(np.minimum(np.linalg.norm(X - [1, 0], axis=1), np.linalg.norm(X + [1, 0], axis=1)) < .15):.3f}")
check("TSGD started on the saddle's stable manifold still leaves it (Theorem 3)", near_saddle == 0)
save_json("apidopoulos2026taming", "data.json", {"rows": r(rows, 4), "p": p_, "q": q_, "sigma": SIG})
done()
