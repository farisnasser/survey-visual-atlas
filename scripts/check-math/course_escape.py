"""Crash course stop 6 (course/escape): escape time near a strict saddle, the stable set (measure zero), perturbed GD,
negative curvature from gradient differences (power iteration), Newton vs saddle-free Newton vs cubic regularization."""
import math
from common import *

print("course_escape")
M32 = 0xFFFFFFFF


def mulberry(seed):
    """Exact port of src/lib/math.js rng (mulberry32 + Box-Muller with a spare)."""
    st = {"a": seed & M32, "spare": None}

    def nxt():
        st["a"] = (st["a"] + 0x6D2B79F5) & M32
        t = st["a"]
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296

    def normal():
        if st["spare"] is not None:
            s = st["spare"]; st["spare"] = None; return s
        u = 0.0
        while u == 0:
            u = nxt()
        v = nxt()
        r = math.sqrt(-2 * math.log(u))
        st["spare"] = r * math.sin(2 * math.pi * v)
        return r * math.cos(2 * math.pi * v)
    nxt.normal = normal
    return nxt


# ---- 1. escape time on f = x^2/2 + y^4/4 - y^2/2 (Lee et al. Sec. 3), eta = 0.1 (visual 'escape')
fl = lambda p: .5 * p[0] ** 2 + .25 * p[1] ** 4 - .5 * p[1] ** 2
gl = lambda p: np.array([p[0], p[1] ** 3 - p[1]])
check_grad("Lee et al.'s example", fl, gl, rand_pts(20, -1.6, 1.6))
ETA = .1


def esc(y0, x0=1.0, eta=ETA, c=.5):
    p = np.array([x0, y0]); t = 0
    while abs(p[1]) <= c and t < 10 ** 6:
        p = p - eta * gl(p); t += 1
    return t
E = {d: esc(d) for d in [1e-2, 1e-4, 1e-6, 1e-8, 1e-12]}
P = {d: math.log(.5 / d) / math.log(1 + ETA) for d in E}
print("  steps until |y| > 0.5 from (1, delta):", E, " prediction ln(0.5/delta)/ln(1.1):", {d: round(v, 1) for d, v in P.items()})
check("text: delta = 1e-2, 1e-4, 1e-6, 1e-8, 1e-12 -> 43, 91, 139, 188, 284 steps", list(E.values()) == [43, 91, 139, 188, 284])
check("each factor 100 in delta costs ln(100)/ln(1.1) = 48.3 steps", np.isclose(math.log(100) / math.log(1.1), 48.3, atol=.05) and all(abs((E[a] - E[b]) - 48.3) < 1.5 for a, b in [(1e-4, 1e-2), (1e-6, 1e-4), (1e-8, 1e-6)]))
check("escape time is within 2 steps of the linear prediction ln(0.5/delta)/ln(1.1)", all(abs(E[d] - P[d]) < 2.1 for d in E))
# weak negative curvature: f = x^2/2 - (gamma/2) y^2 + y^4/4 from y0 = 1e-3, eta = 0.1
def esc_g(gamma, y0=1e-3, eta=ETA):
    y, t = y0, 0
    while abs(y) <= .5 * math.sqrt(gamma) and t < 10 ** 7:
        y = y - eta * (-gamma * y + y ** 3); t += 1
    return t
W = {g_: esc_g(g_) for g_ in [1, .1, .01]}
print("  weak curvature gamma -> steps to |y| > sqrt(gamma)/2 from 1e-3:", W)
check("text: gamma = 1, 0.1, 0.01 -> 67, 524, 4058 steps (about 1/(eta gamma) times a log)", list(W.values()) == [67, 524, 4058], str(W))

# ---- 2. stable set of a bent saddle (visual 'stable'): f = y^2/2 + cos(u) - 1, u = x - 0.25 y^2 (Jin et al.'s example with the axes
# swapped so the picture is wide, then bent)
K = .25
fb = lambda p: p[1] ** 2 / 2 + np.cos(p[0] - K * p[1] ** 2) - 1
def gb(p):
    u = p[0] - K * p[1] ** 2
    return np.array([-np.sin(u), p[1] + 2 * K * p[1] * np.sin(u)])
check_grad("bent saddle", fb, gb, rand_pts(20, -2.4, 2.4))
check_hess_at("bent saddle at 0", fb, [0, 0], [[-1, 0], [0, 1]])
check("critical points in the box: saddle (0, 0), minima (+-pi, 0)", np.linalg.norm(gb([0, 0])) == 0 and all(np.linalg.norm(gb([s * np.pi, 0])) < 1e-12 for s in (1, -1)) and np.isclose(fb([np.pi, 0]), -2))
Rb = np.random.default_rng(4)
L_box = max(np.linalg.norm(fd_hess(fb, p), 2) for p in np.column_stack([Rb.uniform(-4, 4, 4000), Rb.uniform(-2.4, 2.4, 4000)]))
print(f"  gradient-Lipschitz constant on the box |x| <= 4, |y| <= 2.4 (sampled): {L_box:.2f} -> 1/L = {1 / L_box:.3f}")
check("eta = 0.2 < 1/L on the drawn box (Lee et al. Theorem 4 needs a step below 1/L)", .2 < 1 / L_box and L_box < 3.2)
ETB = .2


def run_b(p, T=300):
    p = np.array(p, float)
    for _ in range(T):
        p = p - ETB * gb(p)
    return p
def side(p):
    q = run_b(p); return int(np.sign(q[0] - K * q[1] ** 2))
def xstar(y):
    lo, hi = K * y * y - 1.5, K * y * y + 1.5; s0 = side([lo, y])
    for _ in range(55):
        mid = (lo + hi) / 2
        if side([mid, y]) == s0: lo = mid
        else: hi = mid
    return (lo + hi) / 2
XS = {y: xstar(y) for y in [.5, 1., 1.5, 2.]}
print("  stable curve x*(y):", {y: round(v, 4) for y, v in XS.items()}, " valley floor 0.25 y^2 at y = 2: 1.0")
check("the stable curve is bent: x*(2) = 0.398, neither the eigen-line (0) nor the valley floor (1.0)", round(XS[2.], 3) == .398)
check("it is tangent to the y-axis at the saddle: |x*(0.5)| < 0.1 * 0.5", abs(XS[.5]) < .05)
r = mulberry(11); at_saddle = 0; ends = {1: 0, -1: 0}
for _ in range(2000):
    p = np.array([r() * 8 - 4, r() * 4.8 - 2.4]); q = run_b(p, 600)
    if np.linalg.norm(q) < 1e-3: at_saddle += 1
    else: ends[int(np.sign(q[0]))] += 1
print(f"  2000 random starts in the box: {at_saddle} at the saddle, {ends} at the minima (+-pi, 0)")
check("2000 random starts: none end at the saddle", at_saddle == 0 and ends[1] + ends[-1] == 2000)
def esc_b(p, T=5000):
    p = np.array(p, float); t = 0
    while fb(p) > -1 and t < T:
        p = p - ETB * gb(p); t += 1
    return t, run_b(p, 400)
EB = {}
for dl in [1e-2, 1e-4, 1e-6]:
    t, q = esc_b([XS[2.] + dl, 2.]); EB[dl] = t
    print(f"  start (x*(2) + {dl}, 2): f < -1 after {t} steps, ends near ({q[0]:.3f}, {q[1]:.3f})")
check("starts just beside the stable curve take longer the closer they are (log(1/delta) growth)", EB[1e-6] > EB[1e-4] > EB[1e-2])
t0, q0 = esc_b([.4, 2.])
print(f"  visual default start (0.4, 2): f < -1 after {t0} steps, ends at ({q0[0]:.3f}, {q0[1]:.3f})")
check("visual 'stable' default start (0.4, 2): reaches f < -1 after 39 steps, at the minimum (pi, 0)", t0 == 39 and np.allclose(q0, [np.pi, 0], atol=1e-6))

# ---- 3. perturbed GD (Jin et al. 2017, Algorithm 2) on f = x^2/2 + cos y - 1 (visual 'pgd'), exact port of the JS
fj = lambda p: .5 * p[0] ** 2 + math.cos(p[1]) - 1
gj = lambda p: np.array([p[0], -math.sin(p[1])])
check_grad("Jin et al.'s example", lambda p: fj(p), gj, rand_pts(20, -3, 3))
check_hess_at("Jin example saddle", lambda p: fj(p), [0, 0], [[1, 0], [0, -1]])
ETJ, GTH, TTH, FTH = .2, 1e-3, 80, .05


def pgd(r, seed, x0=(1.5, 0.), T=2000):
    rn = mulberry(seed); x = np.array(x0, float); t_noise = -TTH - 1; xt = None; kicks = []; t = 0
    while t < T:
        if np.linalg.norm(gj(x)) <= GTH and t - t_noise > TTH:
            xt = x.copy(); t_noise = t
            a, b = rn.normal(), rn.normal(); n = math.hypot(a, b); s = r * math.sqrt(rn())
            x = xt + np.array([a / n * s, b / n * s]); kicks.append((t, x[1] - xt[1]))
        if xt is not None and t - t_noise == TTH and fj(x) - fj(xt) > -FTH:
            return xt, kicks, t
        x = x - ETJ * gj(x); t += 1
    return x, kicks, T
xg = gd(gj, [1.5, 0.], ETJ, 400)[-1]
check("plain GD from (1.5, 0) stops at the saddle (f = 0)", np.linalg.norm(xg) < 1e-12 and fj(xg) == 0)
for rr_ in [.1, .01, .001]:
    xe, kk, te = pgd(rr_, 1)
    print(f"  PGD r = {rr_}: kicks at steps {[k[0] for k in kk]} (y-offsets {[f'{k[1]:.1e}' for k in kk]}), ends f = {fj(xe):.4f} at t = {te}")
xe, kk, te = pgd(.1, 1)
check("PGD default (r = 0.1, seed 1): first kick at step 33, second kick at the minimum at step 114, stops at step 194 with f = -2", [k[0] for k in kk] == [33, 114] and te == 194 and abs(fj(xe) + 2) < 1e-6, f"{kk}, {te}, {fj(xe)}")
# escape steps after the kick ~ ln(1/|offset|)/ln(1 + eta)
def steps_after_kick(r, seed):
    xe, kk, te = pgd(r, seed); return kk[1][0] - kk[0][0] if len(kk) > 1 else None
sa = {rr_: steps_after_kick(rr_, 1) for rr_ in [.1, .01, .001]}
print("  steps from the first kick to the second (at the minimum):", sa)
check("smaller kicks take longer, by about ln(10)/ln(1.2) = 12.6 steps per factor 10", sa[.01] > sa[.1] and sa[.001] > sa[.01])

# ---- 5. negative curvature from gradient differences = power iteration on I - eta H (visual 'power')
D_ = 30


def spectrum(gamma):
    return np.concatenate([[-gamma], np.linspace(0, 1, D_ - 1)])


def power(gamma, eta=1., rad=1e-3, seed=5, T=5000):
    lam = spectrum(gamma); rn = mulberry(seed)
    v = np.array([rn.normal() for _ in range(D_)]); v /= np.linalg.norm(v)
    grad = lambda x: lam * x + x ** 3          # f = 1/2 sum lam_i x_i^2 + 1/4 sum x_i^4, saddle at 0
    for t in range(1, T + 1):
        hv = (grad(rad * v) - grad(np.zeros(D_))) / rad
        v = v - eta * hv; v /= np.linalg.norm(v)
        if v @ (lam * v) <= -gamma / 2:
            return t
    return None
check_grad("power-iteration test function (d = 30)", lambda x: .5 * np.sum(spectrum(.1) * x * x) + .25 * np.sum(x ** 4), lambda x: spectrum(.1) * x + x ** 3, rand_pts(5, -1, 1, dim=D_))
PW = {g_: power(g_) for g_ in [1, .1, .01, .001]}
print("  iterations until v^T H v <= -gamma/2 (d = 30, eta = 1, r = 1e-3):", PW)
check("text: gamma = 1, 0.1, 0.01, 0.001 -> 5, 24, 195, 1939 iterations (about 1/gamma)", list(PW.values()) == [5, 24, 195, 1939], str(PW))

# ---- 6. Newton, saddle-free Newton, cubic regularization, GD on f = x^2/2 - y^2/2 + y^4/4 + 0.3 x y^2 (Nesterov-Polyak toy)
C3 = .3
fn = lambda p: .5 * p[0] ** 2 - .5 * p[1] ** 2 + .25 * p[1] ** 4 + C3 * p[0] * p[1] ** 2
gn = lambda p: np.array([p[0] + C3 * p[1] ** 2, -p[1] + p[1] ** 3 + 2 * C3 * p[0] * p[1]])
Hn = lambda p: np.array([[1, 2 * C3 * p[1]], [2 * C3 * p[1], -1 + 3 * p[1] ** 2 + 2 * C3 * p[0]]])
check_grad("Nesterov-Polyak toy", fn, gn, rand_pts(20, -1.6, 1.6))
check_hess_at("Nesterov-Polyak toy at (0.4, -0.7)", fn, [.4, -.7], Hn([.4, -.7]), tol=3e-3)


def cubic_step(g, H, M):
    lam, V = np.linalg.eigh(H); gb_ = V.T @ g
    lo = max(0, -2 * lam[0] / M) + 1e-14
    nr = lambda r_: np.linalg.norm(gb_ / (lam + M * r_ / 2))
    if nr(lo) < lo:
        Db = np.where(lam + M * lo / 2 > 1e-12, -gb_ / (lam + M * lo / 2), 0); Db[0] = np.sqrt(max(lo ** 2 - Db @ Db, 0))
        return V @ Db
    hi = lo + 1
    while nr(hi) > hi:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if nr(mid) > mid else (lo, mid)
    return V @ (-gb_ / (lam + M * hi / 2))


X0 = np.array([1.2, .03])
def run(kind, T=2000):
    x = X0.copy()
    for t in range(1, T + 1):
        g = gn(x)
        if kind == "gd":
            x = x - g / 8
        elif kind == "newton":
            x = x - np.linalg.solve(Hn(x), g)
        elif kind == "sfn":
            lam, V = np.linalg.eigh(Hn(x)); x = x - V @ ((V.T @ g) / np.maximum(np.abs(lam), .1))
        else:
            x = x + cubic_step(g, Hn(x), 10.)
        if np.linalg.norm(gn(x)) < 1e-6:
            return t, x
    return None, x
res = {k: run(k) for k in ["newton", "sfn", "cr", "gd"]}
for k, (t, x) in res.items():
    print(f"  {k}: |grad| < 1e-6 after {t} steps at {np.round(x, 4)}, f = {fn(x):.4f}, lambda_min = {eig(Hn(x))[0]:.3f}")
ys = math.sqrt(1 / (1 - 2 * C3 * C3))
check("Newton converges to the saddle (0, 0) in 4 steps", res["newton"][0] == 4 and np.linalg.norm(res["newton"][1]) < 1e-6)
check("cubic regularization (M = 10) reaches a minimum in 12 steps; GD (eta = 1/8) in 167", res["cr"][0] == 12 and res["gd"][0] == 167 and abs(abs(res["cr"][1][1]) - ys) < 1e-6)
check("saddle-free Newton (|lambda| floored at 0.1) reaches a minimum in 7 steps", res["sfn"][0] == 7 and abs(abs(res["sfn"][1][1]) - ys) < 1e-6, str(res["sfn"][0]))
check("minimum value f = -0.3049 at (-0.366, +-1.104)", round(fn(res["cr"][1]), 4) == -.3049 and np.allclose(np.abs(res["cr"][1]), [.3659, 1.1043], atol=1e-4), f"{fn(res['cr'][1]):.4f} {res['cr'][1]}")
# start exactly on the saddle's stable line y = 0: Newton and GD converge to the saddle, CR still leaves (Lemma 6)
X0 = np.array([1.2, 0.])
res0 = {k: run(k) for k in ["newton", "cr", "gd"]}
print("  from (1.2, 0):", {k: (t, np.round(x, 4).tolist()) for k, (t, x) in res0.items()})
check("from (1.2, 0): Newton and GD end at the saddle, CR reaches a minimum (cannot be attracted, Lemma 6)",
      np.linalg.norm(res0["newton"][1]) < 1e-6 and np.linalg.norm(res0["gd"][1]) < 1e-6 and abs(abs(res0["cr"][1][1]) - ys) < 1e-6)
X0 = np.array([1.2, .03])
# Dauphin's 5x^2 - y^2: one exact Newton step from anywhere lands on the saddle
p0 = np.array([.9, .12]); HA = np.diag([10., -2])
check("Newton on 5x^2 - y^2: one step from (0.9, 0.12) lands exactly on the saddle", np.allclose(p0 - np.linalg.solve(HA, np.array([10 * p0[0], -2 * p0[1]])), 0))
check("SFN step on 5x^2 - y^2 moves y away: (0.9, 0.12) -> (0, 0.24)", np.allclose(p0 - np.linalg.solve(np.abs(HA), np.array([10 * p0[0], -2 * p0[1]])), [0, .24]))
done()
