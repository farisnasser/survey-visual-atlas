"""Jin, Netrapalli, Ge, Kakade, Jordan 2021 (J. ACM 68(2); arXiv 1902.04811). Algorithm 1 (PGD): x <- x - eta (grad f(x) + xi),
xi ~ N(0, (r^2/d) I) at EVERY step (unlike the 2017 PGD that perturbs only when the gradient is small). Theorem 13: an eps-second-order
stationary point within O~(l (f(x0) - f*) / eps^2) iterations, polylogarithmic in d. Algorithm 2 (PSGD) / Theorem 16: stochastic version.
Toy: f(x) = -g x1^2/2 + x1^4/4 + sum_{i>=2} lam x_i^2/2 in R^d, start exactly at the strict saddle 0. The coordinates decouple, so the
escape time is that of x1 driven by N(0, r^2/d) noise: it grows like log d, not polynomially.
"""
from common import *

print("jin2021nonconvex")
GAM, LAM, ETA, RAD = 1., 1., .1, .01
f2 = lambda p: -GAM * p[0] ** 2 / 2 + p[0] ** 4 / 4 + LAM * p[1] ** 2 / 2
g2 = lambda p: np.array([-GAM * p[0] + p[0] ** 3, LAM * p[1]])
check_grad("f (d = 2)", f2, g2, rand_pts(20, -1.5, 1.5))
check_hess_at("at 0 = diag(-g, lam): strict saddle", f2, [0, 0], [[-1, 0], [0, 1]])
check("plain GD from exactly 0 never moves (the gradient is 0)", np.all(g2(np.zeros(2)) == 0))
def escape(d, runs=400, T=20000, seed=0):
    R = np.random.default_rng(seed); x = np.zeros(runs); t_esc = np.full(runs, -1)
    for t in range(T):
        x = x - ETA * (-GAM * x + x ** 3 + R.normal(0, RAD / np.sqrt(d), runs))
        t_esc = np.where((t_esc < 0) & (np.abs(x) > .5), t + 1, t_esc)
        if np.all(t_esc > 0): break
    return t_esc
ds = [1, 4, 16, 64, 256, 1024, 4096, 16384]
med = []
for d in ds:
    te = escape(d, seed=d); med.append([d, float(np.median(te)), float(np.percentile(te, 10)), float(np.percentile(te, 90))])
    print(f"  d = {d:5d}: escape step median {med[-1][1]:.0f} (10-90%: {med[-1][2]:.0f}-{med[-1][3]:.0f})")
m = np.array(med); sl = np.polyfit(np.log(m[:, 0]), m[:, 1], 1)[0]
print(f"  slope of median escape time per unit log d: {sl:.2f}  (theory 1/(2 log(1 + eta g)) = {1 / (2 * np.log(1 + ETA * GAM)):.2f})")
check("escape time grows linearly in log d (slope within 15% of 1/(2 log(1 + eta g)))", abs(sl / (1 / (2 * np.log(1 + ETA * GAM))) - 1) < .15)
check("from d = 1 to d = 16384 the median escape time grows by less than a factor 3", m[-1, 1] < 3 * m[0, 1])
save_json("jin2021nonconvex", "data.json", {"med": med, "eta": ETA, "r": RAD, "slope": sl})

# in-page notes (papers/jin2021nonconvex/notes.md): toy saddle (Sec. 2), escape-time formula, primer numbers
eta_j = .1
def esc_steps(v0, r=1., eta=eta_j):
    u, v, t = 1., v0, 0
    while abs(v) < r:
        u, v = (1 - eta) * u, (1 + eta) * v; t += 1
    return t
for v0 in [1e-3, 1e-6, 1e-9]:
    T_pred = np.log(1 / v0) / np.log(1 + eta_j)
    assert abs(esc_steps(v0) - T_pred) <= 1, (v0, esc_steps(v0), T_pred)
check("notes: T_esc ~ log(r/|delta|)/log(1 + eta) on f = u^2/2 - v^2/2 (to within one step)", True)
check("primer: eta = 0.1, r = 1: ~72 steps from 1e-3, ~217 from 1e-9", abs(np.log(1e3) / np.log(1.1) - 72.5) < .1 and abs(np.log(1e9) / np.log(1.1) - 217.4) < .1)
xi_j = np.random.default_rng(0).normal(0, 1 / np.sqrt(1e4), (2000, 10000))
check("primer: N(0, r^2/d I) has E|xi|^2 = r^2 (r = 1, d = 1e4)", abs(np.mean(np.sum(xi_j ** 2, 1)) - 1) < .005)
comp = xi_j[:, 17]
check("primer: component along a fixed direction has std r/sqrt(d) = 0.01", abs(comp.std() - .01) < .0005)
zc = np.random.default_rng(1).normal(0, .01, 2000000)
check("primer: P(|component| < 1e-4) ~ 0.008", abs(np.mean(np.abs(zc) < 1e-4) - .008) < 3e-4)
check("primer: 100x smaller start costs log(100)/log(1.1) ~ 48 extra steps", abs(np.log(100) / np.log(1.1) - 48) < .5)
u_, v_ = .5, 0.
rngS = np.random.default_rng(2)
for t in range(1000):                                  # noise only along u: v stays exactly 0
    u_, v_ = u_ - eta_j * (u_ + rngS.normal()), v_ - eta_j * (-v_)
check("primer: noise only along u never moves v off 0 (stuck on the saddle's stable line)", v_ == 0)
check("primer: union bound K p = 1000 x 1e-6 = 1e-3", abs(1000 * 1e-6 - 1e-3) < 1e-15)
done()
