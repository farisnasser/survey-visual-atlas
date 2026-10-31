"""Crash course stop 8 (course/noise): finite sums, minibatch gradients, SGD's noise floor, Robbins-Monro schedules,
variance reduction (SVRG), and noise at a saddle. The page's random runs use the mulberry32 RNG of src/lib/math.js; it is
ported exactly below, so every random readout on the page is reproduced here number for number."""
import math

from common import *


class Mulberry:
    def __init__(self, seed=1):
        self.a = seed & 0xFFFFFFFF
        self.spare = None

    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.a
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    def normal(self):
        if self.spare is not None:
            s, self.spare = self.spare, None
            return s
        u = 0.
        while u == 0: u = self()
        v = self()
        r = math.sqrt(-2 * math.log(u))
        self.spare = r * math.sin(2 * math.pi * v)
        return r * math.cos(2 * math.pi * v)


def imul(a, b):
    return (a * b) & 0xFFFFFFFF


# ---- the dataset: a line fit with n = 50 points (same formula in viz.js)
N = 50
X = [4 * i / (N - 1) for i in range(N)]
Y = [.8 * X[i] - .5 + .6 * math.sin(2.7 * i + 1) for i in range(N)]


def loss(w, b):
    return sum((w * X[i] + b - Y[i]) ** 2 for i in range(N)) / N


def grad_i(w, b, i):
    r = w * X[i] + b - Y[i]
    return 2 * r * X[i], 2 * r


def full_grad(w, b):
    gw = gb = 0.
    for i in range(N):
        a, c = grad_i(w, b, i); gw += a; gb += c
    return gw / N, gb / N


def batch_grad(w, b, B, r):
    """B examples drawn independently and uniformly (with replacement), the page's sampler"""
    gw = gb = 0.
    for _ in range(B):
        i = math.floor(r() * N)
        a, c = grad_i(w, b, i); gw += a; gb += c
    return gw / B, gb / B


def sgd_run(start, eta_fn, B, seed, T):
    r = Mulberry(seed); w, b = start; out = [(w, b)]
    for t in range(T):
        gw, gb = batch_grad(w, b, B, r); e = eta_fn(t)
        w, b = w - e * gw, b - e * gb; out.append((w, b))
    return out


A_ = np.stack([np.array(X), np.ones(N)], 1)
W_STAR, B_STAR = np.linalg.lstsq(A_, np.array(Y), rcond=None)[0]
F_STAR = loss(W_STAR, B_STAR)


print("course_noise")
r0 = Mulberry(1)
check("mulberry32 port: seed 1 gives 0.6270739406, 0.0027357212, 0.5274470400 (the JS values)",
      [round(r0(), 10) for _ in range(3)] == [.6270739406, .0027357212, .52744704])
fN = lambda z: loss(z[0], z[1]); gN = lambda z: np.array(full_grad(z[0], z[1]))
check_grad("line-fit finite sum (n = 50)", fN, gN, rand_pts(8, -1, 2))
H = 2 * A_.T @ A_ / N; ev = np.linalg.eigvalsh(H)
print(f"  x* = ({W_STAR:.4f}, {B_STAR:.4f}), f* = {F_STAR:.4f}, Hessian eigenvalues {ev.round(3)}, kappa {ev[1] / ev[0]:.1f}")
check("minimum w* = 0.800, b* = -0.488, f* = 0.180", round(W_STAR, 3) == .8 and round(B_STAR, 3) == -.488 and round(F_STAR, 3) == .18)
check("curvatures 0.450 and 12.33, kappa = 27.4 (one step size must respect the steep one)", round(ev[0], 3) == .45 and round(ev[1], 2) == 12.33 and round(ev[1] / ev[0], 1) == 27.4)

# ---- section 2: unbiased minibatch gradients, variance sigma^2 / B
P0 = (1.8, -1.2)
g0 = np.array(full_grad(*P0)); Gi = np.array([grad_i(*P0, i) for i in range(N)])
s2p = float(np.mean(((Gi - g0) ** 2).sum(1)))
check("unbiased: the average of the n example gradients is the full gradient", np.allclose(Gi.mean(0), g0))
check("at (1.8, -1.2): grad = (7.93, 2.58), sigma^2 = 75.25, sigma^2/5 = 15.05", np.allclose(g0.round(2), [7.93, 2.58]) and round(s2p, 2) == 75.25 and round(s2p / 5, 2) == 15.05)
# exact variance of the mean of B draws: with replacement sigma^2 / B; without replacement (n - B)/(n - 1) sigma^2 / B
import itertools
idx = list(range(6)); sub = Gi[:6]; m6 = sub.mean(0); s6 = np.mean(((sub - m6) ** 2).sum(1))
with_r = np.mean([((sub[list(c)].mean(0) - m6) ** 2).sum() for c in itertools.product(idx, repeat=2)])
wo_r = np.mean([((sub[list(c)].mean(0) - m6) ** 2).sum() for c in itertools.combinations(idx, 2)])
check("enumeration (n = 6, B = 2): with replacement sigma^2/B, without (n - B)/(n - 1) sigma^2/B", np.isclose(with_r, s6 / 2) and np.isclose(wo_r, (6 - 2) / 5 * s6 / 2))
r = Mulberry(1); S = np.array([batch_grad(*P0, 5, r) for _ in range(400)])
mS, vS = S.mean(0), float(np.mean(((S - g0) ** 2).sum(1)))
print(f"  page default (400 minibatches of B = 5, seed 1): mean ({mS[0]:.2f}, {mS[1]:.2f}), spread {vS:.2f}")
check("cloud readout: mean (7.77, 2.56), measured spread 15.02 vs exact 15.05", np.allclose(mS.round(2), [7.77, 2.56]) and round(vS, 2) == 15.02)

# ---- section 3: noise floor. 1-D derivation and the line fit
a_, s_, e_ = 2., 1., .05
V = e_ ** 2 * s_ ** 2 / (1 - (1 - e_ * a_) ** 2)
check("1-D: stationary variance eta^2 s^2/(1 - (1 - eta a)^2) = eta s^2/(a(2 - eta a)); E f = eta s^2/(2(2 - eta a))",
      np.isclose(V, e_ * s_ ** 2 / (a_ * (2 - e_ * a_))) and np.isclose(a_ * V / 2, e_ * s_ ** 2 / (2 * (2 - e_ * a_))))
Gs = np.array([grad_i(W_STAR, B_STAR, i) for i in range(N)]); S2 = float(np.mean((Gs ** 2).sum(1)))
check("sigma*^2 (spread of example gradients at the minimum) = 4.81", round(S2, 2) == 4.81 and np.allclose(Gs.mean(0), 0, atol=1e-12))


def floor_runs(eta, B, T=4000, tau=200):
    c = sgd_run(P0, lambda t: eta, B, 11, T); d = sgd_run(P0, lambda t: eta / (1 + t / tau), B, 12, T)
    w, b = P0; gd = [loss(w, b)]
    for t in range(T): gw, gb = full_grad(w, b); w, b = w - eta * gw, b - eta * gb; gd.append(loss(w, b))
    av = lambda xs: float(np.mean(xs[-1000:]))
    return av([loss(*q) - F_STAR for q in c]), av([loss(*q) - F_STAR for q in d]), av([v - F_STAR for v in gd])


fl = {}
for eta, B in ((.02, 1), (.02, 5), (.005, 1)):
    fl[(eta, B)] = floor_runs(eta, B)
    print(f"  floor eta = {eta}, B = {B}: constant {fl[(eta, B)][0]:.5f} (small-step theory {eta * S2 / (4 * B):.5f}), decaying {fl[(eta, B)][1]:.6f}, GD {fl[(eta, B)][2]:.2e}")
check("eta = 0.005, B = 1: measured floor 0.0062, small-step theory eta sigma*^2/4 = 0.0060", round(fl[(.005, 1)][0], 4) == .0062 and round(.005 * S2 / 4, 4) == .006)
check("eta = 0.02, B = 1 (page default): constant 0.0314 (theory 0.0241), decaying 0.00079",
      round(fl[(.02, 1)][0], 4) == .0314 and round(.02 * S2 / 4, 4) == .0241 and round(fl[(.02, 1)][1], 5) == .00079)
check("eta = 0.02, B = 5: floor 0.0059 (theory 0.0048)", round(fl[(.02, 5)][0], 4) == .0059 and round(.02 * S2 / 20, 4) == .0048)
check("GD with the same step has no floor", fl[(.02, 1)][2] < 1e-9)

# ---- section 4: Robbins-Monro on f = x^2/2 with gradient x + xi, xi ~ N(0, 1), x0 = 2
SCHED = {"const": lambda t: .1, "rm": lambda t: 1 / (t + 2), "fast": lambda t: 1 / (t + 2) ** 2}


def exact_ms(fn, T, x0=2.):
    m = x0 * x0; out = {}
    for t in range(T):
        e = fn(t); m = (1 - e) ** 2 * m + e * e
        if t + 1 in (100, 10000): out[t + 1] = m
    return out


def mc_ms(fn, seed, R=100, T=10000, x0=2.):
    r = Mulberry(seed); xs = [x0] * R; out = {}
    for t in range(T):
        e = fn(t)
        for k in range(R): xs[k] = xs[k] - e * (xs[k] + r.normal())
        if t + 1 in (100, 10000): out[t + 1] = sum(x * x for x in xs) / R
    return out


ex = {k: exact_ms(f, 10000) for k, f in SCHED.items()}
check("E x_t^2 recursion (1 - eta)^2 m + eta^2; constant 0.1 -> eta/(2 - eta) = 0.0526", round(ex["const"][10000], 4) == .0526 and np.isclose(ex["const"][10000], .1 / 1.9))
check("eta_t = 1/(t + 2): E x_t^2 = (x0^2 + t)/(t + 1)^2: 0.0102 at t = 100, 0.000100 at 10^4",
      np.isclose(ex["rm"][100], 104 / 101 ** 2) and np.isclose(ex["rm"][10000], 10004 / 10001 ** 2) and round(ex["rm"][100], 4) == .0102)
xs_, e_s = 2., np.random.default_rng(3).normal(size=50)
for t in range(50): xs_ = xs_ - 1 / (t + 2) * (xs_ + e_s[t])
check("eta_t = 1/(t + 2): x_t = (x0 - sum of the noise)/(t + 1) exactly", np.isclose(xs_, (2 - e_s.sum()) / 51))
check("eta_t = 1/(t + 2)^2: steps sum to pi^2/6 - 1 = 0.645, E x^2 stalls at 1.04",
      np.isclose(sum(1 / (t + 2) ** 2 for t in range(10 ** 6)), np.pi ** 2 / 6 - 1, atol=1e-5) and round(ex["fast"][10000], 2) == 1.04)
check("product of (1 - 1/k^2) over k >= 2 is 1/2 (the start is only halved)", np.isclose(np.prod([1 - 1 / k ** 2 for k in range(2, 200001)]), .5, atol=1e-5))
mc = {k: mc_ms(f, s) for (k, f), s in zip(SCHED.items(), (5, 6, 7))}
print("  schedules, 100 runs, t = 10^4: " + ", ".join(f"{k} {mc[k][10000]:.3g} (exact {ex[k][10000]:.3g})" for k in SCHED))
check("schedules readout (seeds 5, 6, 7) within 25% of the exact values", all(abs(mc[k][10000] / ex[k][10000] - 1) < .25 for k in SCHED))

# ---- section 6: variance reduction (SVRG) vs SGD vs GD, measured in epochs (n gradient evaluations)
L_ = float(ev[1])


def svrg(eta, outer, seed, m=N):
    """one outer loop: full gradient at the snapshot (n evaluations) + m inner steps (2 evaluations each)"""
    r = Mulberry(seed); w, b = P0; out = [loss(w, b) - F_STAR]
    for e in range(outer):
        sw, sb = w, b; mw, mb = full_grad(w, b)
        for k in range(m):
            i = math.floor(r() * N); a, c = grad_i(w, b, i); a0, c0 = grad_i(sw, sb, i)
            w, b = w - eta * (a - a0 + mw), b - eta * (c - c0 + mb)
        out.append(loss(w, b) - F_STAR)
    return out


sv = svrg(.03, 20, 42)   # 20 outer loops x 3 epochs = 60 epochs
gdv = [loss(*P0) - F_STAR]; w, b = P0
for t in range(60): gw, gb = full_grad(w, b); w, b = w - gw / L_, b - gb / L_; gdv.append(loss(w, b) - F_STAR)
sgr = sgd_run(P0, lambda t: .03, 1, 41, 60 * N)
print(f"  after 60 epochs: SVRG {sv[20]:.1e}, GD (1/L) {gdv[60]:.1e}, SGD (0.03) {loss(*sgr[-1]) - F_STAR:.1e}")
check("60 epochs of gradients: SVRG (eta 0.03, m = n) below 1e-12, GD with 1/L 2.7e-3, SGD (0.03) 4.1e-2",
      sv[20] < 1e-12 and round(gdv[60], 4) == .0027 and round(loss(*sgr[-1]) - F_STAR, 3) == .041)
check("SVRG estimator is unbiased: averaging over i gives the full gradient",
      np.allclose(np.mean([np.array(grad_i(1., 0., i)) - np.array(grad_i(.5, .2, i)) for i in range(N)], 0) + np.array(full_grad(.5, .2)), full_grad(1., 0.)))

# ---- section 5: orders of growth, eps = 1e-2, n = 1e6 (constants aside)
n6, e2 = 1e6, 1e-2
check("n = 10^6, eps = 10^-2: GD n eps^-2 = 10^10; SGD eps^-4 = 10^8; SVRG n + n^(2/3) eps^-2 = 1.01e8; SPIDER n + n^(1/2) eps^-2 = 1.1e7, online eps^-3 = 10^6",
      np.isclose(n6 / e2 ** 2, 1e10) and np.isclose(e2 ** -4, 1e8) and np.isclose(n6 + n6 ** (2 / 3) * e2 ** -2, 1.01e8) and np.isclose(n6 + n6 ** .5 * e2 ** -2, 1.1e7) and np.isclose(e2 ** -3, 1e6))
# Ghadimi-Lan Corollary 2.2 with D~ = D_f: E||grad||^2 <= 2 L Delta / N + 2 sigma sqrt(2 L Delta) / sqrt(N)
Ng = lambda eps: (2 * np.sqrt(2) / eps ** 2) ** 2   # the noise term alone
check("Ghadimi-Lan with noise: the sigma/sqrt(N) term needs N ~ eps^-4 (halving eps: x16)", np.isclose(Ng(.005) / Ng(.01), 16))
Lg, Dl, sg_, ep = 2., 3., .5, 1e-2   # with D~ = D_f: E||grad||^2 <= 2 L Delta / N + 2 sigma sqrt(2 L Delta) / sqrt(N)
Nn = 8 * Lg * Dl * sg_ ** 2 / ep ** 4
check("Corollary 2.2 with D~ = D_f: the noise term is <= eps^2 exactly when N >= 8 L Delta sigma^2 / eps^4", np.isclose(2 * sg_ * np.sqrt(2 * Lg * Dl) / np.sqrt(Nn), ep ** 2)
      and np.isclose(Lg * (Lg * (np.sqrt(2 * Dl / Lg)) ** 2 / Nn + 2 * np.sqrt(2 * Dl / Lg) * sg_ / np.sqrt(Nn)), 2 * Lg * Dl / Nn + 2 * sg_ * np.sqrt(2 * Lg * Dl) / np.sqrt(Nn)))

# ---- section 7: noise at a saddle, f = x^2/2 + y^4/4 - y^2/2, components f +- s y (or +- s x), eta = 0.1, start (1.5, 0)
fS = lambda z: .5 * z[0] ** 2 + .25 * z[1] ** 4 - .5 * z[1] ** 2; gS = lambda z: np.array([z[0], z[1] ** 3 - z[1]])
check_grad("saddle landscape", fS, gS, rand_pts(8, -1.5, 1.5))
check("saddle at 0 with curvatures 1 and -1; minima (0, +-1) with f = -1/4", np.allclose(eig(fd_hess(fS, [0., 0.])), [-1, 1], atol=1e-3) and np.isclose(fS([0, 1]), -.25))


def esc(mode, s, seed=31, eta=.1, T=600):
    r = Mulberry(seed); x, y = 1.5, 0.
    for t in range(T):
        gx, gy = x, y ** 3 - y
        if mode != "gd":
            sg2 = s if r() < .5 else -s
            if mode == "y": gy += sg2
            else: gx += sg2
        x, y = x - eta * gx, y - eta * gy
        if abs(y) > .5: return t + 1
    return None


check("noise along y (s = 0.3, seed 31): escapes (|y| > 0.5) after 23 steps; GD and noise along x never leave y = 0",
      esc("y", .3) == 23 and esc("x", .3) is None and esc("gd", .3) is None)
check("heavy tails, Prop. 5: eta_crit = 2b/(sigma^2 (d + b + 1)) = 2/3 for d = b = sigma^2 = 1", np.isclose(2 * 1 / (1 * (1 + 1 + 1)), 2 / 3))
check("epoch: n = 50, B = 5 -> 10 steps per epoch", 50 // 5 == 10)
done()
