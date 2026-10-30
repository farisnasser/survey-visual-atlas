"""Daneshmand, Kohler, Lucchi, Hofmann 2018: Correlated Negative Curvature (CNC) and CNC-PGD / CNC-SGD (Alg. 1, 2).

Finite sum around a strict saddle: f = x^2/2 - y^2/2 + y^4/4,  f_i = f + a_i y + c_i x  (zero-mean a, c).
CNC at the saddle: E <e_y, grad f_i(0)>^2 = mean(a_i^2) > 0.  Without CNC (a = 0) no stochastic gradient points along e_y.
"""
from common import *

print("daneshmand2018escaping")
n = 64
R0 = np.random.default_rng(2)
a = R0.normal(0, 1, n); c = R0.normal(0, 1, n); a -= a.mean(); c -= c.mean()
f = lambda p: .5 * p[0] ** 2 - .5 * p[1] ** 2 + .25 * p[1] ** 4
g = lambda p: np.array([p[0], -p[1] + p[1] ** 3])
check_grad("f", f, g, rand_pts(20, -1.5, 1.5))
check("CNC constant at the saddle: E<v, grad f_i>^2 = mean(a^2)", abs(np.mean(a ** 2) - np.mean([(a[i]) ** 2 for i in range(n)])) < 1e-15, f"gamma = {np.mean(a ** 2):.3f}")


def run(method, cnc, seed, T=3000, eta=.05, r=.4, g_thres=1e-6, t_thres=50):
    R = np.random.default_rng(seed); w = np.array([1.5, 0.0]); tn = -t_thres - 1
    gi = lambda w, i: g(w) + np.array([c[i], a[i] if cnc else 0.0])
    for t in range(T):
        if method == "gd":
            w = w - eta * g(w)
        elif method == "pgd":
            if np.sum(g(w) ** 2) <= g_thres and t - tn >= t_thres:
                tn = t; w = w - r * gi(w, R.integers(0, n))
            else:
                w = w - eta * g(w)
        else:  # cnc-sgd
            step = r if t % t_thres == 0 else eta
            w = w - step * gi(w, R.integers(0, n))
    return w


for cnc in (True, False):
    for m in ("gd", "pgd", "sgd"):
        ends = np.array([run(m, cnc, s) for s in range(20)])
        esc = np.mean(np.abs(ends[:, 1]) > .5)
        print(f"  CNC={cnc} {m}: fraction escaped {esc:.2f}")
        if m == "gd":
            check(f"GD from the stable set stays at the saddle", esc == 0)
        elif cnc:
            check(f"{m} with CNC noise escapes every time", esc == 1)
        else:
            check(f"{m} without CNC (noise orthogonal to e_y) never escapes", esc == 0)


# escape time vs dimension: one kick of energy r^2 at the saddle of -w1^2/2 + w1^4/4 + |w_rest|^2/2, then GD
def esc_time(d, kind, seed, r=.1, eta=.1):
    R = np.random.default_rng(seed)
    if kind == "iso":
        u = R.normal(size=d); xi = r * u / np.linalg.norm(u)
    else:  # CNC: the same energy, a fixed share along the escape direction (E<e1, xi>^2 = r^2/2)
        u = R.normal(size=d - 1); xi = np.concatenate([[r / np.sqrt(2) * R.choice([-1, 1])], r / np.sqrt(2) * u / np.linalg.norm(u)])
    w = xi.copy()
    for t in range(100000):
        if abs(w[0]) > .5:
            return t
        w = w - eta * np.concatenate([[-w[0] + w[0] ** 3], w[1:]])
    return 100000


rows = []
for d in [2, 10, 100, 1000, 10000]:
    ti = np.median([esc_time(d, "iso", s) for s in range(60)]); tc = np.median([esc_time(d, "cnc", s) for s in range(60)])
    rows.append([d, float(ti), float(tc)])
    print(f"  d = {d}: isotropic kick {ti:.0f} steps, CNC-aligned {tc:.0f}")
check("isotropic kick: escape time grows with d (projection ~ r/sqrt(d))", rows[-1][1] > rows[0][1] + 20)
check("CNC-aligned noise: escape time independent of d", max(r_[2] for r_ in rows) - min(r_[2] for r_ in rows) <= 2)
save_json("daneshmand2018escaping", "data.json", {"a": r(a, 5), "c": r(c, 5), "dim": rows})
done()
