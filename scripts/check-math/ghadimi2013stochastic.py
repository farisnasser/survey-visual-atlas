"""Ghadimi & Lan 2013: RSG on a smooth non-convex landscape; empirical E||grad f(x_R)||^2 vs Corollary 2.2."""
from common import *

print("ghadimi2013stochastic")
# globally L-smooth, bounded below: quadratic + Gaussian wells and one bump
WELLS = [(-1.3, 1.0, .45, .5), (-.9, -1.1, -.7, .35), (.7, -.1, 1.2, .3)]  # (amp, cx, cy, s2): amp<0 is a well


def f(p):
    x, y = p
    v = .12 * (x * x + y * y)
    for a, cx, cy, s2 in WELLS:
        v += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s2)
    return v


def g(p):
    x, y = p
    gx, gy = .24 * x, .24 * y
    for a, cx, cy, s2 in WELLS:
        e = a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s2)
        gx += e * (-2 * (x - cx) / s2)
        gy += e * (-2 * (y - cy) / s2)
    return np.array([gx, gy])


check_grad("f", f, g, rand_pts(60, -2, 2))
xs = np.linspace(-2.2, 2.2, 221)
G = np.array([[f([x, y]) for x in xs] for y in xs])
f_star = G.min()
Lmax = 0
for x in xs[::6]:
    for y in xs[::6]:
        Lmax = max(Lmax, np.max(np.abs(eig(fd_hess(f, [x, y])))))
print(f"  f* ~ {f_star:.4f}, L ~ {Lmax:.3f} (max |eig H| on the grid)")
# critical points by Newton from a grid of starts
crit = []
for s in rand_pts(400, -2, 2, seed=9):
    p = np.array(s)
    for _ in range(60):
        H = fd_hess(f, p)
        try:
            p = p - np.linalg.solve(H, g(p))
        except np.linalg.LinAlgError:
            break
    if np.linalg.norm(g(p)) < 1e-9 and np.max(np.abs(p)) < 2.2 and not any(np.linalg.norm(p - c[0]) < 1e-4 for c in crit):
        ev = eig(fd_hess(f, p))
        kind = "min" if ev[0] > 0 else ("max" if ev[1] < 0 else "saddle")
        crit.append((p, kind, f(p)))
for p, k, v in sorted(crit, key=lambda c: c[2]):
    print(f"  critical {k:6s} at ({p[0]: .3f},{p[1]: .3f}) f={v: .4f}")
kinds = [c[1] for c in crit]
check("landscape has minima, saddles and a maximum", kinds.count("min") >= 2 and "saddle" in kinds and "max" in kinds)
# RSG with Gaussian gradient noise: per-coordinate std s -> sigma^2 = 2 s^2
s_noise = .6
sigma = np.sqrt(2) * s_noise
x1 = np.array([1.6, 1.5])
Df = np.sqrt(2 * (f(x1) - f_star) / Lmax)
rng = np.random.default_rng(0)
rows = []
for N in [10, 30, 100, 300, 1000, 3000]:
    Dt = Df
    gam = min(1 / Lmax, Dt / (sigma * np.sqrt(N)))
    vals = []
    for run in range(300):
        x = x1.copy()
        R = rng.integers(1, N + 1)  # constant step -> P_R uniform (Eq. 2.3)
        for k in range(1, R):
            x = x - gam * (g(x) + rng.normal(0, s_noise, 2))
        vals.append(np.sum(g(x) ** 2))
    emp = np.mean(vals)
    bound = Lmax * (Lmax * Df ** 2 / N + (Dt + Df ** 2 / Dt) * sigma / np.sqrt(N))
    rows.append((N, emp, bound))
    print(f"  N={N:5d}: step {gam:.4f}  E||grad||^2 = {emp:.4f}  <=  L*B_N = {bound:.4f}")
check("empirical E||grad f(x_R)||^2 below the Corollary 2.2 bound for every N", all(e <= b for _, e, b in rows))
check("empirical value decreases with N", rows[-1][1] < rows[0][1])
# zeroth-order estimator (Eq. 3.12) is unbiased for grad f_mu
mu, p = .05, np.array([.3, -.4])
U = rng.normal(size=(200000, 2))
est = np.mean([(f(p + mu * u) - f(p)) / mu * u for u in U[:200000:4]], axis=0)
check("RSGF estimator ~ grad f for small mu", np.linalg.norm(est - g(p)) < .03, f"{np.round(est, 3)} vs {np.round(g(p), 3)}")

# in-page notes (papers/ghadimi2013stochastic/notes.md): the notes' smooth-SGD bound and the primer's numbers
# notes Sec. 3: (1/N) sum E||grad f(w_k)||^2 <= 2 Delta/(eta N) + L eta sigma^2 on a smooth quadratic with Gaussian noise
Aq = np.diag([2.0, 0.5])           # f(w) = w^T A w / 2, L = 2, f_inf = 0
Lq, sq, eta_q, Nq = 2.0, .5, .3, 50                       # eta <= 1/L
w1 = np.array([1.0, -2.0]); Dq = .5 * w1 @ Aq @ w1
acc = np.zeros(Nq)
for run in range(4000):
    w = w1.copy()
    for k in range(Nq):
        acc[k] += np.sum((Aq @ w) ** 2)
        w = w - eta_q * (Aq @ w + rng.normal(0, sq, 2))
avg = acc.mean() / 4000
rhs = 2 * Dq / (eta_q * Nq) + Lq * eta_q * 2 * sq ** 2
check("notes Sec. 3 bound holds on a noisy quadratic", avg <= rhs, f"{avg:.4f} <= {rhs:.4f}")
# primer idea 1: sin w, L = 1, from 0 to 0.5
check("primer: descent-lemma bound 0.625 >= sin 0.5 ~ 0.479", abs((0 + .5 + .125) - .625) < 1e-12 and round(np.sin(.5), 3) == .479)
# primer idea 2: G in {1,3}: E G = 2, E G^2 = 5 = 4 + 1
check("primer: E[G]=2, E[G^2]=5=2^2+1", np.mean([1, 3]) == 2 and np.mean([1, 9]) == 5)
# primer idea 4: eta* = sqrt(2 Delta/(N L s2)), bound = 2 sqrt(2 Delta L s2 / N)
for N, e_exp, b_exp in [(100, .141, .283), (10000, .0141, .0283)]:
    es = np.sqrt(2 / N); b = 2 / (es * N) + es
    grid = min(2 / (e * N) + e for e in np.linspace(es / 3, 3 * es, 20001))
    check(f"primer: N={N}: eta* ~ {e_exp}, bound ~ {b_exp} (minimum of 2D/(eta N)+L eta s2)",
          abs(es - e_exp) < 6e-4 * (e_exp / .141) and abs(b - b_exp) < 6e-4 * (b_exp / .283) and abs(b - 2 * np.sqrt(2 / N)) < 1e-12 and abs(grid - b) < 1e-6 and es <= 1)
# primer idea 5: average of 0.9, 0.3, 0.1, 0.1
check("primer: uniform pick of (0.9,0.3,0.1,0.1) has mean 0.35", abs(np.mean([.9, .3, .1, .1]) - .35) < 1e-12)
done()
