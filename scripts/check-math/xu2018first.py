"""Xu, Jin, Yang 2018: NEON (Alg. 1) = GD on the centered function, NEON+ = Nesterov on it (zeta = 1 - sqrt(eta gamma)).

f_hat_x(u) = f(x + u) - f(x) - grad f(x)^T u,   u_{t+1} = u_t - eta (grad f(x + u_t) - grad f(x)).
"""
from common import *

print("xu2018first")
f = lambda p: .5 * p[0] ** 2 - .3 * p[1] ** 2 + .25 * p[1] ** 4 + .4 * p[0] * p[1] ** 2
g = lambda p: np.array([p[0] + .4 * p[1] ** 2, -.6 * p[1] + p[1] ** 3 + .8 * p[0] * p[1]])
check_grad("f", f, g, rand_pts(20, -1.5, 1.5))
x0 = np.array([.15, .05])
fh = lambda u: f(x0 + u) - f(x0) - g(x0) @ u
gh = lambda u: g(x0 + u) - g(x0)
check_grad("f_hat", fh, gh, rand_pts(20, -1, 1))
check("f_hat(0) = 0 and grad f_hat(0) = 0 although grad f(x) != 0", abs(fh(np.zeros(2))) < 1e-15 and np.allclose(gh(np.zeros(2)), 0) and np.linalg.norm(g(x0)) > .1)
Hx = fd_hess(f, x0); lm = eig(Hx)[0]
print(f"  hess f(x) eigenvalues {np.round(eig(Hx), 4)}")
check("x has negative curvature", lm < -.4)
# NEON in 2D from a random point on the sphere of radius r
R = np.random.default_rng(0)
eta, rr, F = .1, .01, .002
a = R.uniform(0, 2 * np.pi); u = rr * np.array([np.cos(a), np.sin(a)])
for t in range(400):
    u = u - eta * gh(u)
    if fh(u) <= -2.5 * F:
        break
ev, V = np.linalg.eigh(Hx)
check("2D NEON: detection, and u aligns with the negative eigenvector", fh(u) <= -2.5 * F and abs(u @ V[:, 0]) / np.linalg.norm(u) > .9, f"t = {t}, cos = {abs(u @ V[:, 0]) / np.linalg.norm(u):.3f}")

# d = 50 quadratic: f_hat(u) = 1/2 u^T H u, H = diag(-gamma, spread [0, 1]); iterations until f_hat <= -thr
d = 50


def detect(gamma, plus, seed, eta=.5, thr=1e-3, r0=1e-3, T=20000):
    Rr = np.random.default_rng(seed)
    Hd = np.concatenate([[-gamma], np.linspace(0, 1, d - 1)])
    u = Rr.normal(size=d); u *= r0 / np.linalg.norm(u); yp = u.copy(); zeta = 1 - np.sqrt(eta * gamma)
    for t in range(T):
        if .5 * np.sum(Hd * u * u) <= -thr:
            return t
        if plus:
            y = u - eta * Hd * u; u = y + zeta * (y - yp); yp = y
        else:
            u = u - eta * Hd * u
    return T


gams = [.1, .03, .01, .003, .001]
tn = [np.median([detect(gm, False, s) for s in range(15)]) for gm in gams]
tp = [np.median([detect(gm, True, s) for s in range(15)]) for gm in gams]
for gm, a_, b_ in zip(gams, tn, tp):
    print(f"  gamma = {gm}: NEON {a_:.0f} iterations, NEON+ {b_:.0f}")
sn = np.polyfit(np.log(gams), np.log(tn), 1)[0]; sp = np.polyfit(np.log(gams), np.log(tp), 1)[0]
check("NEON iterations ~ 1/gamma (power method)", abs(sn + 1) < .15, f"slope {sn:.3f}")
check("NEON+ iterations ~ 1/sqrt(gamma) (accelerated)", abs(sp + .5) < .15, f"slope {sp:.3f}")
save_json("xu2018first", "data.json", {"gams": gams, "neon": tn, "neonp": tp, "slopes": [round(sn, 3), round(sp, 3)]})
done()
