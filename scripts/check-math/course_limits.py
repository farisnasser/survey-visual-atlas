"""Crash course stop 12 (course/limits): oracles, upper vs lower bounds, Carmon et al.'s zero chain and its scaling,
the probability-p chain of Arjevani et al., NP-hardness (a quartic whose Hessian is blind), and the complexity ladder.
Every number on the page is recomputed here. RNG = a port of src/lib/math.js rng (mulberry32)."""
import math
from math import erf, sqrt, e, pi
from common import *
np.seterr(over="ignore", invalid="ignore")

print("course_limits")


def rng(seed):
    a = [seed & 0xffffffff]
    imul = lambda x, y: ((x & 0xffffffff) * (y & 0xffffffff)) & 0xffffffff

    def nxt():
        a[0] = (a[0] + 0x6d2b79f5) & 0xffffffff; t = a[0]
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296
    return nxt


# ---------- 1. the upper bound of stop 3 ----------
check("text: GD guarantee 2 L (f(x0) - f*)/eps^2 = 20,000 for L = 1, gap 1, eps = 0.01", np.isclose(2 / .01 ** 2, 2e4))

# ---------- 2. Carmon et al.'s hard chain f_T ----------
Psi = lambda t: 0.0 if t <= .5 else math.exp(1 - 1 / (2 * t - 1) ** 2)
dPsi = lambda t: 0.0 if t <= .5 else Psi(t) * 4 / (2 * t - 1) ** 3
Phi = lambda t: sqrt(e) * sqrt(2 * pi) * .5 * (1 + erf(t / sqrt(2)))
dPhi = lambda t: sqrt(e) * math.exp(-t * t / 2)


def fT(x):
    s = -Psi(1) * Phi(x[0])
    for i in range(1, len(x)): s += Psi(-x[i - 1]) * Phi(-x[i]) - Psi(x[i - 1]) * Phi(x[i])
    return s


def gT(x):
    T = len(x); G = np.zeros(T)
    G[0] = -Psi(1) * dPhi(x[0])
    for i in range(1, T):
        G[i] += -Psi(-x[i - 1]) * dPhi(-x[i]) - Psi(x[i - 1]) * dPhi(x[i])
        G[i - 1] += -dPsi(-x[i - 1]) * Phi(-x[i]) - dPsi(x[i - 1]) * Phi(x[i])
    return G


check_grad("f_T (T = 4)", fT, gT, rand_pts(20, -2, 2, dim=4))
check("Psi(1) = 1, Psi = 0 on (-inf, 1/2]; Phi' = sqrt(e) exp(-t^2/2), Phi'(0) = sqrt(e) = 1.649", Psi(1) == 1 and Psi(.5) == 0 and Psi(-3) == 0 and round(dPhi(0), 3) == 1.649)
check("Phi matches its integral definition", abs(Phi(.3) - sqrt(e) * np.trapezoid(np.exp(-np.linspace(-12, .3, 200001) ** 2 / 2), np.linspace(-12, .3, 200001))) < 1e-6)
R = np.random.default_rng(12); ok = True
for _ in range(300):
    T = 8; i = int(R.integers(1, T)); x = np.zeros(T); x[:i - 1] = R.normal(0, 2, i - 1); ok &= bool(np.all(gT(x)[i:] == 0))
check("zero chain: if x_j = 0 for j >= i, then the gradient is 0 in coordinates j > i", ok)
check("Lemma 1: Psi(x) Phi'(y) > 1 for x >= 1, |y| < 1", min(Psi(x) * dPhi(y) for x in np.linspace(1, 4, 31) for y in np.linspace(-.999, .999, 41)) > 1)
viol = sum(1 for _ in range(3000) for x in [R.normal(0, 1.5, 5)] if np.any(np.abs(x) < 1) and np.linalg.norm(gT(x)) <= 1)
check("Lemma 2: |grad f_T(x)| > 1 whenever some |x_i| < 1 (3000 random points, T = 5)", viol == 0)
check("Lemma 3 (sampled, T = 10): f_T(0) - inf f_T <= 12 T and |grad f_T| <= 23 sqrt(T)",
      fT(np.zeros(10)) - min(fT(R.normal(0, 3, 10)) for _ in range(2000)) <= 120 and max(np.linalg.norm(gT(R.normal(0, 2, 10))) for _ in range(2000)) <= 23 * sqrt(10))


def gd_chain(T, eta, kmax=100000):
    """GD from 0; returns (step when |grad| <= 1, step when x_T first moves, x_1 at the first move of x_2)."""
    x = np.zeros(T); hit = first2 = None; x1at = None; allnz = None
    for k in range(kmax):
        G = gT(x)
        if allnz is None and np.all(x != 0): allnz = k
        if hit is None and np.linalg.norm(G) <= 1: hit = k
        if hit is not None and allnz is not None: break
        xn = x - eta * G
        if T >= 2 and first2 is None and xn[1] != 0: first2, x1at = k + 1, x[0]
        x = xn
    return hit, allnz, first2, x1at


h2, a2, f2, x1 = gd_chain(2, .05)
print(f"  T = 2, eta = 0.05: |grad| <= 1 at step {h2}; x_2 first non-zero after step {f2}, when x_1 = {x1:.3f}")
check("visual 'chain2': T = 2, eta 0.05: x_2 stays exactly 0 for 7 steps and moves at step 8, once x_1 = 0.554 > 1/2", f2 == 8 and round(x1, 3) == .554)
check("visual 'chain2': |grad f_2| <= 1 first at step 29", h2 == 29)
for eta in np.round(np.arange(.01, .1001, .005), 3):
    h, a_, f_, x1_ = gd_chain(2, eta)
    assert x1_ > .5 and h is not None, eta
check("visual 'chain2': for every slider eta in [0.01, 0.1], x_2 starts moving only after x_1 > 1/2, and GD reaches |grad| <= 1", True)
# path stays in the drawn box [-0.6, 3]^2
ok = True
for e_ in np.round(np.arange(.01, .1001, .005), 3):
    P_ = gd(gT, [0, 0], e_, gd_chain(2, e_)[0]); ok &= bool(P_.min() > -.6 and P_.max() < 3)
check("visual 'chain2': for every slider eta, the GD path until |grad| <= 1 (where the ball stops) stays in the drawn box [-0.6, 3]^2", ok)


# ---------- 3./4. visual 'chain': GD (p = 1) and SGD with the probability-p oracle (Arjevani et al. Eq. 17) ----------
def prog(x, a):
    idx = np.nonzero(np.abs(x) > a)[0]
    return int(idx[-1] + 1) if len(idx) else 0


def chain_run(T, p, eta=.05, kmax=20000):
    r = rng(1000 * T + int(round(100 * p))); x = np.zeros(T); disc = None
    for k in range(kmax):
        if disc is None and prog(x, 0) >= T: disc = k
        if disc is not None and np.linalg.norm(gT(x)) <= 1: return disc, k
        G = gT(x); z = 1.0 if p >= 1 else (1.0 if r() < p else 0.0)
        if p < 1:
            j = prog(x, .25); G[j:] *= z / p
        x = x - eta * G
    return disc, None


TAB = {(T, p): chain_run(T, p) for T in range(2, 17) for p in (1.0, .2, .05)}
for p in (1.0, .2, .05):
    print(f"  p = {p}: queries until all T coordinates are non-zero / until |grad| <= 1:", {T: TAB[(T, p)] for T in (2, 4, 8, 12, 16)})
check("visual 'chain': every run (T = 2..16, p = 1, 0.2, 0.05) discovers all coordinates and reaches |grad| <= 1", all(v[0] is not None and v[1] is not None for v in TAB.values()))
K1 = [TAB[(T, 1.0)][1] for T in range(2, 17)]
slope = np.polyfit(np.arange(2, 17), K1, 1)[0]
check("visual 'chain': exact GD needs a number of steps growing linearly in T (about 6.5 per coordinate)", np.all(np.diff(K1) > 0) and round(slope, 1) == 6.5, f"slope {slope:.2f}")
check("text: T = 8: exact GD reaches |grad| <= 1 after 71 queries; p = 0.2 after 90; p = 0.05 after 191",
      [TAB[(8, p)][1] for p in (1.0, .2, .05)] == [71, 90, 191], f"{[TAB[(8, p)][1] for p in (1.0, .2, .05)]}")
check("text: T = 16: 121 / 156 / 529 queries", [TAB[(16, p)][1] for p in (1.0, .2, .05)] == [121, 156, 529], f"{[TAB[(16, p)][1] for p in (1.0, .2, .05)]}")
# unbiasedness and variance of the oracle (Lemma 3 of Arjevani et al.)
xr = np.r_[R.normal(0, 2, 3), np.zeros(3)]
def g_noisy(x, z, p):
    G = gT(x).copy(); j = prog(x, .25); G[j:] *= z / p; return G
samples = [g_noisy(xr, float(R.random() < .2), .2) for _ in range(100000)]
check("oracle: unbiased (mean of 1e5 samples = grad f_T)", np.allclose(np.mean(samples, 0), gT(xr), atol=.03 * (1 + np.abs(gT(xr)).max())))
check("oracle: variance <= 23^2 (1 - p)/p", np.mean([np.sum((s - gT(xr)) ** 2) for s in samples]) <= 23 ** 2 * .8 / .2)

# ---------- 3. scaling: f(x) = (L lam^2 / l1) f_T(x / lam) ----------
L, l1, eps = 2.0, 7.0, .05; lam = l1 * eps / L
fs = lambda x: L * lam ** 2 / l1 * fT(x / lam); gs_ = lambda x: L * lam / l1 * gT(x / lam)
check_grad("scaled chain", fs, gs_, rand_pts(10, -.2, .2, dim=3))
xx = rand_pts(5, -.1, .1, dim=3)
check("scaling: |grad f(x)| = eps |grad f_T(x/lam)| with lam = l1 eps / L, so |grad f| > eps unless every |x_i| >= lam",
      all(np.isclose(np.linalg.norm(gs_(p)), eps * np.linalg.norm(gT(p / lam))) for p in xx))
H1, H2 = fd_hess(fs, np.array([.01, .02, 0.])), fd_hess(fT, np.array([.01, .02, 0.]) / lam)
check("scaling: Hessian of f = (L / l1) x Hessian of f_T, so f is L-smooth when f_T is l1-smooth", np.allclose(H1, L / l1 * H2, atol=1e-4))
check("scaling: gap <= (L lam^2 / l1) 12 T = 12 T l1 eps^2 / L, so T = Delta L / (12 l1 eps^2) fits a gap Delta",
      np.isclose(L * lam ** 2 / l1 * 12, 12 * l1 * eps ** 2 / L))
check("text: (p + 1)/p = 2, 3/2, 4/3; at eps = 0.01: 1e4, 1e3, 464", [(p + 1) / p for p in (1, 2, 3)] == [2, 1.5, 4 / 3] and round(.01 ** -2) == 10000 and round(.01 ** -1.5) == 1000 and round(.01 ** (-4 / 3)) == 464)
check("text (stochastic): variance 23^2 eps^2 (1 - p)/p <= sigma^2 needs p ~ eps^2/sigma^2; T/p ~ (Delta L/eps^2)(sigma^2/eps^2)",
      np.isclose((1 / eps ** 2) * (1 / eps ** 2), eps ** -4))

# ---------- 5. a quartic whose Hessian is blind: f(u) = u1^4 + 2 q u1^2 u2^2 + u2^4 ----------
qf = lambda u, q: u[0] ** 4 + 2 * q * u[0] ** 2 * u[1] ** 2 + u[1] ** 4
qg = lambda u, q: np.array([4 * u[0] ** 3 + 4 * q * u[0] * u[1] ** 2, 4 * u[1] ** 3 + 4 * q * u[0] ** 2 * u[1]])
for q in (-1.5, -1, 0, 1.3):
    check_grad(f"quartic q = {q}", lambda u: qf(u, q), lambda u: qg(u, q), rand_pts(6, -1, 1))
check("quartic: gradient and Hessian at 0 are zero for every q", all(np.allclose(fd_hess(lambda u: qf(u, q), [0, 0]), 0, atol=1e-6) for q in (-2, -1, 0, 1.5)))
th = np.linspace(0, 2 * np.pi, 100001)
for q in (-2, -1.2, -1, 0, 1, 1.5):
    m = min(qf((np.cos(t), np.sin(t)), q) for t in th[::50]); assert np.isclose(m, min(1, (1 + q) / 2), atol=1e-6), (q, m)
check("quartic: min of f on the unit circle = (1 + q)/2 for q <= 1 (at the diagonals), 1 for q >= 1", True)
check("quartic: f(r u) = r^4 f(u), so 0 is a local min iff (1 + q)/2 >= 0 iff q >= -1; q = -1.2 gives -0.1 on the circle", np.isclose((1 - 1.2) / 2, -.1))
copos = lambda q: all(np.array([a, b]) @ np.array([[1, q], [q, 1]]) @ np.array([a, b]) >= -1e-12 for a in np.linspace(0, 1, 41) for b in np.linspace(0, 1, 41))
check("quartic: Q = [[1, q], [q, 1]] is copositive iff q >= -1, PSD iff |q| <= 1 (f(u) = v^T Q v with v = (u1^2, u2^2))",
      copos(-1) and not copos(-1.05) and copos(1.5) and np.linalg.eigvalsh([[1, 1.5], [1.5, 1]])[0] < 0 and np.linalg.eigvalsh([[1, -1], [-1, 1]])[0] > -1e-12)
q_ = -1.3
u = np.array([.5, .45]); out = None
for k in range(400):
    u = u - .05 * qg(u, q_)
    if np.abs(u).max() > 1: out = k + 1; break
check("quartic ball: q = -1.3, GD from (0.5, 0.45) leaves the unit square (along the diagonal) after 30 steps", out == 30 and abs(abs(u[0]) - abs(u[1])) < .1 * np.abs(u).max(), f"{out}, {u}")
P = gd(lambda u: qg(u, .5), [.5, .45], .05, 400)
check("quartic ball: q = 0.5, GD from (0.5, 0.45) creeps toward 0 (|u| < 0.25 after 400 steps, still not 0: a slow, degenerate minimum)", .01 < np.linalg.norm(P[-1]) < .25, f"{np.linalg.norm(P[-1]):.3f}")

# ---------- 6. the ladder: exponents a in eps^-a ----------
LAD = {"GD": 2, "PGD": 2, "PAGD": 1.75, "CR": 1.5, "SGD": 4, "SPIDER": 3, "SCR": 3.5, "Fang19": 3.5, "Natasha2": 3.25, "HVP-RVR": 3}
check("ladder: at eps = 1e-3 the counts eps^-a are 1e6 (a = 2), 1.8e5 (1.75), 3.2e4 (1.5), 1e12 (4), 1e9 (3), 3.2e10 (3.5), 5.6e9 (3.25)",
      [f"{1e-3 ** -a:.1e}" for a in (2, 1.75, 1.5, 4, 3, 3.5, 3.25)] == ["1.0e+06", "1.8e+05", "3.2e+04", "1.0e+12", "1.0e+09", "3.2e+10", "5.6e+09"])
check("ladder: kappa log(1/eps) vs sqrt(kappa) log(1/eps): kappa = 100, eps = 1e-3: 691 vs 69", round(100 * math.log(1e3)) == 691 and round(10 * math.log(1e3)) == 69)
done()
