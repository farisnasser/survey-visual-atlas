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

# in-page notes (papers/fang2019sharp/notes.md): notes Secs. 3, 5 and primer numbers (coupled-path k = 25 checked above)
check("notes Sec. 3: ceil(log 10 / log 1.1) = 25 and both starts lie inside B = 0.01", int(np.ceil(np.log(10) / np.log(1.1))) == 25 and np.hypot(.001, .002) < .01)
check("notes Sec. 5: 2^(7/2) ~ 11.31", abs(2 ** 3.5 - 11.31) < 5e-3)
e_ = 1e-3                                     # scales: B^2/(eta K0) ~ eps^1.5, gap/that ~ eps^-1.5 exits, times K0 ~ eps^-2 -> eps^-3.5
B_, et_, K_ = e_ ** .5, e_ ** 1.5, e_ ** -2
check("notes Sec. 5: B = eps^1/2, eta = eps^3/2, K0 = eps^-2 give decrease ~eps^1.5 per exit and eps^-3.5 total", np.isclose(B_ ** 2 / (et_ * K_), e_ ** 1.5) and np.isclose(1 / (B_ ** 2 / (et_ * K_)) * K_, e_ ** -3.5))
check("primer: eps = 0.01 -> eps^-4 = 1e8, eps^-3.5 = 1e7", np.isclose(.01 ** -4, 1e8) and np.isclose(.01 ** -3.5, 1e7))
Lc, sg, ep = 1., 1., .1                       # one-step bound E f+ <= f - eta(1 - L eta/2)|g|^2 + L eta^2 sigma^2/2
eta_ = ep ** 2 / (Lc * sg ** 2)
gain = eta_ * (1 - Lc * eta_ / 2) * ep ** 2 - Lc * eta_ ** 2 * sg ** 2 / 2
check("primer: with eta = eps^2/(L sigma^2) the net per-step gain is ~eps^4/2 (order eps^4)", abs(gain / (ep ** 4 / 2) - 1) < .02, f"{gain:.3e}")
rng_f = np.random.default_rng(3); xi_ = rng_f.standard_normal((200, 10000, 4)) / 2.   # E|xi|^2 = 1 in R^4
avg = np.linalg.norm(xi_.mean(1), axis=1); sm = np.linalg.norm(xi_.sum(1), axis=1)
check("primer: K = 1e4 unit-size noises: average ~0.01, sum ~100 (sqrt K)", abs(np.sqrt(np.mean(avg ** 2)) - .01) < 5e-4 and abs(np.sqrt(np.mean(sm ** 2)) - 100) < 5)
check("primer: eta sigma sqrt(K0) = eps^1/2 sigma when eta = eps^1.5, K0 = eps^-2", np.isclose(et_ * np.sqrt(K_), e_ ** .5))
qerr = abs(-np.sin(.1) - (-.1))
check("primer: cos x at 0.1: quadratic-model gradient error 1.7e-4 <= rho r^2/2 = 0.005", abs(qerr - 1.7e-4) < 5e-6 and qerr <= .005 and abs(-np.sin(.1) + .09983) < 5e-6)
done()
