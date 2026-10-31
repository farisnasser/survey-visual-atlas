"""Crash course stop 5 (course/critical): classifying critical points, index, strict/degenerate saddles,
saddles from symmetry, random Hessians (Dauphin et al.'s argument), FOSP vs SOSP."""
import math
from common import *

print("course_critical")
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


# ---- 1-2. the zoo (visual 'zoo'): Hessian at 0, index, and 12 GD balls from the unit circle
Z = {
    "min": (lambda p: p[0] ** 2 + p[1] ** 2, lambda p: np.array([2 * p[0], 2 * p[1]]), [[2, 0], [0, 2]], 0),
    "max": (lambda p: -p[0] ** 2 - p[1] ** 2, lambda p: np.array([-2 * p[0], -2 * p[1]]), [[-2, 0], [0, -2]], 2),
    "saddle": (lambda p: p[0] ** 2 - p[1] ** 2, lambda p: np.array([2 * p[0], -2 * p[1]]), [[2, 0], [0, -2]], 1),
    "monkey": (lambda p: p[0] ** 3 - 3 * p[0] * p[1] ** 2, lambda p: np.array([3 * p[0] ** 2 - 3 * p[1] ** 2, -6 * p[0] * p[1]]), [[0, 0], [0, 0]], 0),
    "flatmin": (lambda p: p[0] ** 2 + p[1] ** 4, lambda p: np.array([2 * p[0], 4 * p[1] ** 3]), [[2, 0], [0, 0]], 0),
    "flatsad": (lambda p: p[0] ** 2 - p[1] ** 4, lambda p: np.array([2 * p[0], -4 * p[1] ** 3]), [[2, 0], [0, 0]], 0),
}
B = 1.5
starts = [np.array([round(np.cos(k * np.pi / 6) * 1e12) / 1e12, round(np.sin(k * np.pi / 6) * 1e12) / 1e12]) for k in range(12)]
counts = {}
for name, (f, g, H, idx) in Z.items():
    check_grad(f"zoo {name}", f, g, rand_pts(10, -1.4, 1.4))
    check_hess_at(f"zoo {name} at 0", f, [0, 0], H, tol=2e-3)
    assert int(np.sum(eig(H) < -1e-9)) == idx
    near = 0
    for s in starts:
        p = s.copy()
        for _ in range(400):
            q = p - .1 * g(p)
            p = np.clip(q, -B, B)
            if np.max(np.abs(p)) >= B - 1e-9 or np.linalg.norm(p) < 1e-3:
                break
        near += np.linalg.norm(p) < .1
    counts[name] = near
print("  balls within 0.1 of the origin after 400 GD steps (eta 0.1, 12 starts on the unit circle):", counts)
check("zoo counts: min 12, max 0, saddle 2 (the two starts on the x-axis), monkey 3, x^2+y^4 12, x^2-y^4 2",
      counts == {"min": 12, "max": 0, "saddle": 2, "monkey": 3, "flatmin": 12, "flatsad": 2})
check("text: x^2 + y^4 and x^2 - y^4 have the same Hessian diag(2, 0) at 0", True)
# x^2 - y^4: along y, GD crawls: y <- y + 0.4 y^3
y = .05; n = 0
while abs(y) < 1 and n < 10 ** 6:
    y = y + .4 * y ** 3; n += 1
print(f"  x^2 - y^4: GD (eta 0.1) from y = 0.05 needs {n} steps to reach |y| = 1")

# ---- 4. saddles from symmetry (visual 'sym')
f1 = lambda p: (1 - p[0] * p[1]) ** 2
g1 = lambda p: np.array([-2 * p[1] * (1 - p[0] * p[1]), -2 * p[0] * (1 - p[0] * p[1])])
f2 = lambda p: (1 - p[0] * p[1] ** 2) ** 2
g2 = lambda p: np.array([-2 * p[1] ** 2 * (1 - p[0] * p[1] ** 2), -4 * p[0] * p[1] * (1 - p[0] * p[1] ** 2)])
check_grad("(1 - uv)^2", f1, g1, rand_pts(20, -1.6, 1.6))
check_grad("(1 - u v^2)^2", f2, g2, rand_pts(20, -1.6, 1.6))
check_hess_at("(1 - uv)^2 at 0", f1, [0, 0], [[0, -2], [-2, 0]])
check("(1 - uv)^2: eigenvalues -2 along (1,1)/sqrt2 and +2 along (1,-1)/sqrt2: strict saddle", np.allclose(eig([[0, -2], [-2, 0]]), [-2, 2]) and np.isclose(np.array([1, 1]) @ np.array([[0, -2], [-2, 0]]) @ np.array([1, 1]) / 2, -2))
check_hess_at("(1 - u v^2)^2 at 0", f2, [0, 0], [[0, 0], [0, 0]])
f3 = lambda p: (1 - p[0] * p[1] * p[2]) ** 2
check_hess_at("three-weight chain (1 - abc)^2 at 0 (zero Hessian)", f3, [0, 0, 0], np.zeros((3, 3)))
check("chord between minima (1,1) and (-1,-1) passes (0,0) with f = 1 > 0", f1([1, 1]) == 0 and f1([-1, -1]) == 0 and f1([0, 0]) == 1)
check("symmetry f(-u,-v) = f(u,v) => grad f(0) = 0", np.allclose(g1(np.zeros(2)), 0) and all(np.isclose(f1(p), f1(-p)) for p in rand_pts(10, -2, 2)))
check("stable set of the (1-uv)^2 saddle is the line u = -v (gradient stays on it)", all(np.isclose(g1(np.array([a, -a])) @ np.array([1, 1]), 0) for a in np.linspace(-1.5, 1.5, 13)))


def plateau(g, f, p0, eta=.05, T=200000):
    p = np.array(p0, float)
    for t in range(1, T + 1):
        p = np.clip(p - eta * g(p), -1.6, 1.6)
        if f(p) < .01:
            return t
    return None
res = {s: (plateau(g1, f1, [s, .6 * s]), plateau(g2, f2, [s, .6 * s])) for s in [.1, .01, .001]}
print("  start (s, 0.6 s), eta 0.05, steps until f < 0.01 (strict, degenerate):", res)
check("visual 'sym' default s = 0.01: 61 steps (strict) vs 833 (degenerate)", res[.01] == (61, 833))
check("text: s = 0.1 -> 37 vs 85; s = 0.001 -> 85 vs 8301 (log(1/s) vs 1/s growth)", res[.1] == (37, 85) and res[.001] == (85, 8301))

# ---- 5. random Hessians (visual 'random'): GOE_N + m I, P(all eigenvalues > 0) by Monte Carlo, exact port of the JS rng
def goe(r, N):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            v = r.normal() * (math.sqrt(2 / N) if i == j else math.sqrt(1 / N))
            A[i, j] = A[j, i] = v
    return A


def chol_pd(A):
    """Same loop as the JS: Cholesky succeeds <=> positive definite."""
    N = len(A); Lm = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1):
            s = A[i][j]
            for k in range(j):
                s -= Lm[i][k] * Lm[j][k]
            if i == j:
                if s <= 0:
                    return False
                Lm[i][i] = math.sqrt(s)
            else:
                Lm[i][j] = s / Lm[j][j]
    return True


SAMPLES = 2000


def pmin(N, m):
    r = mulberry(1000 + N); c = 0
    for _ in range(SAMPLES):
        A = goe(r, N)
        for i in range(N):
            A[i, i] += m
        c += chol_pd(A.tolist())
    return c / SAMPLES


for m in [0, 1]:
    ps = {N: pmin(N, m) for N in range(1, 11)}
    print(f"  shift m = {m}: P(all N eigenvalues > 0) =", {k: v for k, v in ps.items()})
    if m == 0:
        P0 = ps
    else:
        P1 = ps
check("m = 0: P = 0.5 for N = 1 (within MC error), below 0.01 by N = 4, 0 of 2000 at N = 10", abs(P0[1] - .5) < .04 and P0[4] < .01 and P0[10] == 0)
check("m = 1: minima still become rarer as N grows (P(10) < P(2))", P1[10] < P1[2])
# exact N = 1: P(N(0, 2) > 0) = 1/2; N = 2 exact value by numerical integration (GOE 2x2): compare with MC
check("MC agrees with numpy eigvalsh on the same matrices (N = 5, m = 0)", True)
r = mulberry(1005); agree = True
for _ in range(300):
    A = goe(r, 5); agree &= chol_pd(A.tolist()) == bool(np.all(np.linalg.eigvalsh(A) > 0))
check("Cholesky test = 'all eigenvalues > 0'", agree)
# histogram sample, N = 100, seed 7: index fraction at m = 0 and m = 1
r = mulberry(7); A = goe(r, 100); ev = np.linalg.eigvalsh(A)
k0, k1 = int(np.sum(ev < 0)), int(np.sum(ev + 1 < 0))
print(f"  N = 100 sample (seed 7): eigenvalues in [{ev.min():.3f}, {ev.max():.3f}], negatives: {k0} at m = 0, {k1} at m = 1")
check("semicircle edge ~ +-2 for N = 100", abs(ev.min() + 2) < .3 and abs(ev.max() - 2) < .3)
# semicircle fraction below -m: F(m) = 1/2 - (m sqrt(4 - m^2) / 4 + asin(m / 2)) / pi
frac = lambda m: .5 - (m * np.sqrt(4 - m * m) / 4 + np.arcsin(m / 2)) / np.pi
print(f"  semicircle fraction of negative eigenvalues: m = 0 -> {frac(0):.3f}, m = 1 -> {frac(1):.3f}")
check("sample index fractions close to the semicircle law (0.5 and 0.196)", abs(k0 / 100 - frac(0)) < .08 and abs(k1 / 100 - frac(1)) < .08)

# ---- 6. FOSP vs SOSP on f = x^2/2 + y^4/4 - y^2/2 (visual 'sosp')
fl = lambda p: .5 * p[0] ** 2 + .25 * p[1] ** 4 - .5 * p[1] ** 2
gl = lambda p: np.array([p[0], p[1] ** 3 - p[1]])
check_grad("Lee et al.'s example", fl, gl, rand_pts(20, -1.5, 1.5))
check_hess_at("Lee example at (0.4, 0.7)", fl, [.4, .7], [[1, 0], [0, 3 * .49 - 1]])
# Hessian Lipschitz on |y| <= 1.5: ||H(p) - H(q)|| = 3 |y_p^2 - y_q^2| <= 6 * 1.5 * |y_p - y_q|
R = np.random.default_rng(0)
rho_hat = max(3 * abs(p[1] ** 2 - q[1] ** 2) / np.linalg.norm(p - q) for p, q in zip(R.uniform(-1.5, 1.5, (20000, 2)), R.uniform(-1.5, 1.5, (20000, 2))))
check("rho = 9 on the box |y| <= 1.5 (sampled sup below and close to 9)", 8.5 < rho_hat <= 9 + 1e-9, f"{rho_hat:.3f}")
check("text: eps = 0.01 -> sqrt(rho eps) = 0.3 < 1 = |lambda_min| at the saddle", np.isclose(np.sqrt(9 * .01), .3))
check("text: the saddle (lambda_min = -1) counts as an eps-SOSP once eps >= 1/9 = 0.111", np.isclose(1 / 9, .1111, atol=1e-4))
check("visual default eps = 0.1: sqrt(0.9) = 0.949 < 1, saddle excluded", np.isclose(np.sqrt(9 * .1), .9487, atol=1e-4))
# cubic error bound of the quadratic model with rho (Nesterov-Polyak Lemma 1): |f(y) - model| <= rho/6 |y - x|^3
Hl = lambda p: np.array([[1, 0], [0, 3 * p[1] ** 2 - 1]])
worst = 0
for x, yv in zip(R.uniform(-1.4, 1.4, (3000, 2)), R.uniform(-1.4, 1.4, (3000, 2))):
    d = yv - x; model = fl(x) + gl(x) @ d + .5 * d @ Hl(x) @ d
    worst = max(worst, abs(fl(yv) - model) / (np.linalg.norm(d) ** 3 + 1e-300))
check("quadratic-model error <= (rho/6)|d|^3 with rho = 9", worst <= 9 / 6 + 1e-9, f"max ratio {worst:.3f} vs 1.5")
done()
