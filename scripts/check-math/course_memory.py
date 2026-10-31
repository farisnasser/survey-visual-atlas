"""Crash course stop 7 (course/memory): heavy ball, EMA view, the 2x2 iteration matrix, Polyak's tuning, Nesterov,
overshoot on the stop-1 curve, Lookahead wrapping the heavy ball."""
from common import *

print("course_memory")
LC = 10.0                                   # sharp curvature (y) of the valley; flat curvature mu = LC / kappa
START = np.array([-1.45, .5])               # stop 3's start


def valley(mu):
    H = np.array([mu, LC])
    return (lambda z: .5 * float((H * z * z).sum())), (lambda z: H * z)


def polyak(L, mu):
    return 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2, ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2


def run(kind, mu, eta, beta=0., tol=1e-8, T=20000):
    """steps until f < tol from START ('div' if it blows up); the same updates as viz.js"""
    f, g = valley(mu)
    z, zp = START.copy(), START.copy()
    for t in range(T):
        if f(z) < tol: return t
        if not np.isfinite(f(z)) or f(z) > 1e6: return "div"
        if kind == "gd": z = z - eta * g(z)
        elif kind == "hb": z, zp = z - eta * g(z) + beta * (z - zp), z
        else: y = z + beta * (z - zp); z, zp = y - eta * g(y), z
    return T


for mu in (1., .1):
    f, g = valley(mu)
    check_grad(f"valley kappa = {LC / mu:.0f}", f, g, rand_pts(10, -1.6, 1.6))

# ---- section 1: GD's zig-zag on stop 3's bowl (kappa = 10)
check("GD best step 2/(L + mu) = 2/11, contraction (kappa - 1)/(kappa + 1) = 9/11 = 0.818", round(9 / 11, 3) == .818)
f, g = valley(1.)
p = [START.copy()]
for _ in range(4): p.append(p[-1] - 2 / 11 * g(p[-1]))
p = np.array(p)
check("GD on the bowl: the y-coordinate flips sign every step, x keeps its sign", np.all(np.sign(p[1:, 1]) != np.sign(p[:-1, 1])) and np.all(p[:, 0] < 0))

# ---- section 3: momentum as an average of past gradients
b = .9
check("beta = 0.9: steady gradient gain 1/(1 - beta) = 10", np.isclose(1 / (1 - b), 10))
check("beta = 0.9: alternating gradient gain 1/(1 + beta) = 0.526", round(1 / (1 + b), 3) == .526)
v = 0.; out = []
for t in range(400): v = b * v + (1. + .6 * (-1) ** t); out.append(v)   # velocity sum without -eta: steady 1, flip 0.6
tail = np.array(out[-2:])
check("unrolled sum: steady part -> 1/(1-beta), alternating part -> 1/(1+beta)", np.isclose(tail.mean(), 10, atol=1e-9) and np.isclose(abs(tail[0] - tail[1]) / 2, .6 / 1.9, atol=1e-9))
check("EMA weights (1 - beta) beta^k sum to 1", np.isclose(sum((1 - b) * b ** k for k in range(5000)), 1))
check("beta = 0.9: the last 7 values carry 52% of the EMA weight", round(1 - b ** 7, 3) == .522)

# ---- section 4: the valley (3D), steps to f < 1e-8
res = {}
for kap, mu in ((10, 1.), (100, .1)):
    eP, bP = polyak(LC, mu); bN = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
    res[kap] = dict(gd=run("gd", mu, 2 / (LC + mu)), hbP=run("hb", mu, eP, bP), nagP=run("nag", mu, eP, bP),
                    hbN=run("hb", mu, 1 / LC, bN), nagN=run("nag", mu, 1 / LC, bN), eP=eP, bP=bP, bN=bN)
    print(f"  kappa = {kap}: Polyak eta = {eP:.4f}, beta = {bP:.4f}; Nesterov eta = 1/L = 0.1, beta = {bN:.4f}; " + str({k: v for k, v in res[kap].items() if k[0] in 'gn' or k.startswith('hb')}))
check("Polyak's tuning, kappa = 10: eta = 0.2309, beta = 0.2699", round(res[10]["eP"], 4) == .2309 and round(res[10]["bP"], 4) == .2699)
check("Polyak's tuning, kappa = 100: eta = 0.3306, beta = 0.6694", round(res[100]["eP"], 4) == .3306 and round(res[100]["bP"], 4) == .6694)
check("Nesterov's tuning beta = (sqrt k - 1)/(sqrt k + 1): 0.519 (kappa 10), 0.818 (kappa 100)", round(res[10]["bN"], 3) == .519 and round(res[100]["bN"], 3) == .818)
check("kappa = 10 steps to f < 1e-8: GD 48, heavy ball (Polyak) 20, Nesterov (its tuning) 31",
      (res[10]["gd"], res[10]["hbP"], res[10]["nagN"]) == (48, 20, 31))
check("kappa = 100 steps to f < 1e-8: GD 469, heavy ball (Polyak) 71, Nesterov (its tuning) 100",
      (res[100]["gd"], res[100]["hbP"], res[100]["nagN"]) == (469, 71, 100))
check("Nesterov with Polyak's heavy-ball tuning diverges (kappa = 10 and 100)", res[10]["nagP"] == "div" and res[100]["nagP"] == "div")
check("heavy ball with Nesterov's tuning: 30 (kappa 10), 84 (kappa 100) steps", (res[10]["hbN"], res[100]["hbN"]) == (30, 84))

# ---- section 5: the 2x2 iteration matrix
def hb_roots(lam, eta, beta): return np.roots([1, -(1 + beta - eta * lam), beta])
def nag_roots(lam, eta, beta): return np.roots([1, -(1 + beta) * (1 - eta * lam), beta * (1 - eta * lam)])


ok = True
for lam, eta, beta in [(1, .1, .9), (10, .05, .5), (.3, .4, .2), (5, .3, .7)]:
    M = np.array([[1 + beta - eta * lam, -beta], [1, 0]])
    ok &= np.allclose(sorted(np.linalg.eigvals(M), key=lambda z: (z.real, z.imag)), sorted(hb_roots(lam, eta, beta), key=lambda z: (z.real, z.imag)))
    u = [1., 1.]
    for _ in range(30): u.append((1 + beta - eta * lam) * u[-1] - beta * u[-2])
    z = np.array([1., 1.]); zp = z.copy(); xs = [1.]
    for _ in range(30): z, zp = z - eta * lam * z + beta * (z - zp), z; xs.append(z[0])
    ok &= np.allclose(u[1:], xs)
check("heavy ball along curvature lambda: u_{t+1} = (1 + beta - eta lambda) u_t - beta u_{t-1}; eigenvalues of M = roots", ok)
check("complex roots have modulus sqrt(beta)", all(np.allclose(abs(hb_roots(l, .1, .64)), .8) for l in np.linspace(1, 30, 12) if (1 + .64 - .1 * l) ** 2 < 4 * .64))
check("complex exactly when (1 - sqrt b)^2 < eta lambda < (1 + sqrt b)^2",
      all(((1 + b_ - e) ** 2 < 4 * b_) == ((1 - np.sqrt(b_)) ** 2 < e < (1 + np.sqrt(b_)) ** 2) for b_ in (.2, .5, .9) for e in np.linspace(.001, 4, 300)))
for kap in (10., 100.):
    mu, L = 1., kap; eP, bP = polyak(L, mu)
    w = max(max(abs(hb_roots(l, eP, bP))) for l in np.linspace(mu, L, 4001))
    gdw = max(abs(1 - 2 / (L + mu) * l) for l in (mu, L))
    check(f"kappa = {kap:.0f}: tuned heavy ball worst |r| = (sqrt k - 1)/(sqrt k + 1) = {(np.sqrt(kap) - 1) / (np.sqrt(kap) + 1):.3f}; GD {gdw:.3f}",
          abs(w - (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)) < 1e-6 and np.isclose(gdw, (kap - 1) / (kap + 1)))
    check(f"kappa = {kap:.0f}: eta mu = (1 - sqrt b)^2 and eta L = (1 + sqrt b)^2 (window edges)", np.isclose(eP * mu, (1 - np.sqrt(bP)) ** 2) and np.isclose(eP * L, (1 + np.sqrt(bP)) ** 2))
    bN = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
    wn = max(max(abs(nag_roots(l, 1 / L, bN))) for l in np.linspace(mu, L, 4001))
    check(f"kappa = {kap:.0f}: Nesterov (eta = 1/L) worst |r| = 1 - 1/sqrt(kappa) = {1 - 1 / np.sqrt(kap):.3f}", abs(wn - (1 - 1 / np.sqrt(kap))) < 1e-6)
check("rates printed in the text: 0.519 / 0.818 (heavy ball), 0.818 / 0.980 (GD), 0.684 / 0.900 (Nesterov)",
      round((np.sqrt(10) - 1) / (np.sqrt(10) + 1), 3) == .519 and round(99 / 101, 3) == .980 and round(1 - 1 / np.sqrt(10), 3) == .684)
n_gd, n_hb = np.log(1e6) / -np.log(99 / 101), np.log(1e6) / -np.log(9 / 11)
check("kappa = 100: a factor 10^6 takes ~691 GD steps vs ~69 heavy-ball steps", round(n_gd) == 691 and round(n_hb) == 69, f"{n_gd:.1f}, {n_hb:.1f}")
n_nag = np.log(1e6) / -np.log(.9)
check("kappa = 100: ... and ~131 Nesterov steps", round(n_nag) == 131, f"{n_nag:.1f}")

# ---- section 6: overshoot on the stop-1 curve
F = lambda t: .1 * t * t + .5 * np.sin(3 * t); dF = lambda t: .2 * t + 1.5 * np.cos(3 * t)
check_grad("stop-1 curve", lambda z: F(z[0]), lambda z: [dF(z[0])], rand_pts(10, -3, 3, 1))


def roll(t0, beta, eta=.05, T=3000):
    x = xp = t0; ups = 0; fp = F(x)
    for k in range(T):
        x, xp = x - eta * dF(x) + beta * (x - xp), x
        if k < 200 and F(x) > fp + 1e-12: ups += 1
        fp = F(x)
    return x, ups


ends = {b_: roll(2.4, b_) for b_ in (0., .9, .95, .99)}
print("  overshoot from 2.4, eta = 0.05:", {k: (round(v[0], 3), v[1]) for k, v in ends.items()})
check("from 2.4: GD (beta = 0) ends in the local minimum 1.50 and f never increases", np.isclose(ends[0.][0], 1.5035, atol=1e-3) and ends[0.][1] == 0)
check("beta = 0.9 also ends at 1.50, but f rises on 92 of the first 200 steps (beta = 0.95: 96)", np.isclose(ends[.9][0], 1.5035, atol=1e-3) and ends[.9][1] == 92 and ends[.95][1] == 96)
check("beta = 0.95 rolls over the bump into the global minimum -0.50", np.isclose(ends[.95][0], -.5013, atol=1e-3))
check("beta = 0.99 rolls past the global minimum into the worst valley -2.50", np.isclose(ends[.99][0], -2.5045, atol=1e-3))
check("losses: -0.264 (1.50), -0.474 (-0.50), 0.156 (-2.50)", round(F(1.50352), 3) == -.264 and round(F(-.5013), 3) == -.474 and round(F(-2.5045), 3) == .156)
check("fourth minimum at 3.5032 just outside stop 1's window (viz constant)", abs(dF(3.5032)) < 1e-3 and .2 - 4.5 * np.sin(3 * 3.5032) > 0 and F(3.5032) > F(1.5035))

# ---- section 7: Jin et al. 2018 vs perturbed GD, eps = 1e-4
check("eps = 1e-4: eps^-2 = 1e8, eps^-7/4 = 1e7 (ten times fewer, constants and logs aside)", np.isclose(1e-4 ** -2, 1e8) and np.isclose(1e-4 ** -1.75, 1e7))

# ---- section 8: Lookahead wrapping the heavy ball (velocity kept across syncs, as in the paper's default)
def la_run(mu, eta, beta, k, alpha, T=20000, tol=1e-8):
    f, g = valley(mu)
    phi = START.copy(); z = phi.copy(); zp = z.copy(); c = 0
    for t in range(T):
        if f(phi) < tol: return t
        z, zp = z - eta * g(z) + beta * (z - zp), z; c += 1
        if c == k:
            vel = z - zp; phi = phi + alpha * (z - phi); z = phi.copy(); zp = z - vel; c = 0
    return T


def la_rate(lam, eta, beta, k, alpha):
    """per-step contraction of Lookahead(heavy ball) along curvature lambda: spectral radius of the (phi, v) cycle map ^ (1/k)"""
    M = np.array([[1 + beta - eta * lam, -beta], [1, 0]]); Mk = np.linalg.matrix_power(M, k)
    cols = []
    for phi, v in ((1., 0.), (0., 1.)):
        uk, ukm1 = Mk @ np.array([phi, phi - v])
        cols.append([phi + alpha * (uk - phi), uk - ukm1])
    return max(abs(np.linalg.eigvals(np.array(cols).T))) ** (1 / k)


mu = .1
hb99, la99 = run("hb", mu, .1, .99), la_run(mu, .1, .99, 20, .5)
rh = max(max(abs(hb_roots(l, .1, .99))) for l in (mu, LC)); rl = max(la_rate(l, .1, .99, 20, .5) for l in (mu, LC))   # the valley has exactly these two curvatures
print(f"  Lookahead, kappa = 100, eta = 0.1, beta = 0.99, k = 20, alpha = 0.5: HB {hb99} steps (rate {rh:.4f}), LA {la99} steps (rate {rl:.4f})")
check("under-damped heavy ball (beta = 0.99): 1425 steps; Lookahead(k = 20, alpha = 0.5) 340", (hb99, la99) == (1425, 340))
check("... per-step rate on the valley's two curvatures: heavy ball 0.9950, Lookahead 0.9759", round(rh, 4) == .995 and round(rl, 4) == .9759)
eP, bP = polyak(LC, mu)
rhP = max(max(abs(hb_roots(l, eP, bP))) for l in (mu, LC)); rlP = max(la_rate(l, eP, bP, 20, .5) for l in (mu, LC))
check(f"with Polyak's tuning Lookahead is slower per step ({rlP:.3f} vs {rhP:.3f})", rlP > rhP and round(rlP, 3) == .980 and round(rhP, 3) == .818)
for x0 in (.1, .5):    # simulation agrees with the cycle map on one coordinate
    lam = LC; z = np.array([x0]); zp = z.copy(); phi = z.copy(); c = 0
    for t in range(200):
        z, zp = z - .1 * lam * z + .99 * (z - zp), z; c += 1
        if c == 20: vel = z - zp; phi = phi + .5 * (z - phi); z = phi.copy(); zp = z - vel; c = 0
    assert abs(phi[0]) < x0 * (la_rate(lam, .1, .99, 20, .5) ** 200) * 50
check("Lookahead simulation decays at the cycle-map rate", True)
done()
