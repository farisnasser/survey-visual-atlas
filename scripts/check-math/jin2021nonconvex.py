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
done()
