"""Fang, Lin, Zhang 2019: SGD with the anchor-ball rule (Algorithm 2) and the coupling argument.

Algorithm 2: x_{k+1} = x_k - eta (grad f(x_k) + xi_k);  if |x_k - x_0| > B: x_0 <- x_k, k <- 0;  after K0 in-ball steps return the mean.
Noise xi uniform in a ball of radius sigma (a dispersive distribution, as in the paper's examples).
"""
from common import *

print("fang2019sharp")
f = lambda p: .25 * (p[0] ** 2 - 1) ** 2 + .5 * p[1] ** 2
g = lambda p: np.array([p[0] ** 3 - p[0], p[1]])
check_grad("f", f, g, rand_pts(20, -1.5, 1.5))
check_hess_at("saddle", f, [0, 0], [[-1, 0], [0, 1]])


def ball(R, s):
    a = R.uniform(0, 2 * np.pi); return s * np.sqrt(R.uniform()) * np.array([np.cos(a), np.sin(a)])


def alg2(seed, eta=.02, sigma=1.0, B=.25, K0=400, T=200000):
    R = np.random.default_rng(seed); x = np.array([0.0, 1.2]); x0 = x.copy(); k = 0; buf = []; exits = 0
    for t in range(T):
        x = x - eta * (g(x) + ball(R, sigma)); k += 1; buf.append(x.copy())
        if np.linalg.norm(x - x0) > B:
            x0 = x.copy(); k = 0; buf = []; exits += 1
        if k >= K0:
            return np.mean(buf[-K0:], 0), exits, t
    return None, exits, T


outs = [alg2(s) for s in range(20)]
check("Algorithm 2 stops in every run", all(o[0] is not None for o in outs), f"median {np.median([o[2] for o in outs]):.0f} steps, {np.median([o[1] for o in outs]):.0f} exits")
check("averaged output is near a minimum (x = +-1): small gradient, positive curvature", all(np.linalg.norm(g(o[0])) < .1 and abs(abs(o[0][0]) - 1) < .1 for o in outs))
x_last = outs[0][0]
print(f"  example output {np.round(x_last, 3)}, |grad f| = {np.linalg.norm(g(x_last)):.3f}")
# averaging certifies the gradient: for a quadratic, grad f(mean) = mean grad f
Hq = np.diag([2., .5]); pts = np.random.default_rng(0).normal(size=(50, 2))
check("quadratic: gradient at the average = average gradient", np.allclose(Hq @ pts.mean(0), (pts @ Hq).mean(0)))
# coupling example from the notes: q = (-u^2 + v^2)/2, eta = 0.1, identical noise
u1, u2 = .001, -.001
k = 0
while abs(u1 - u2) <= .02:
    k += 1; n_ = np.random.default_rng(k).normal()
    u1 = 1.1 * u1 - .1 * n_; u2 = 1.1 * u2 - .1 * n_
check("notes: coupled paths separate as 0.002*1.1^k; first exceed 2B = 0.02 at k = 25", k == 25 and abs(abs(u1 - u2) - .0216694) < 1e-6, f"k = {k}, distance {abs(u1 - u2):.7f}")
done()
