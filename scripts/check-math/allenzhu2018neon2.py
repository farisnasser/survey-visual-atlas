"""Allen-Zhu & Li 2018 (NEON2): Neon2online with same-sample gradient differences (Alg. 2) and Theorem 1's two outcomes.

x_1 = x_0 + xi (tiny Gaussian),  x_{t+1} = x_t - eta (grad f_i(x_t) - grad f_i(x_0)),  i uniform;  stop with
v = (x_{t+1} - x_0)/|.| once |x_{t+1} - x_0| >= r, else return "bottom" after T steps.
Theorem 1 (w.h.p.): bottom => hess f(x_0) >= -delta I;  v => v^T hess v <= -delta/2.
"""
from common import *

print("allenzhu2018neon2")
# 2D finite sum for the 3D tab: f = x^2/2 - (k/2) y^2 + y^4/4 ;  f_i = f + a_i x + b_i y + c_i (x^2 - y^2)/2 + e_i x y
n = 100
R0 = np.random.default_rng(5)
a, b, c, e = [R0.normal(0, s, n) for s in (1, .8, .6, .5)]
a -= a.mean(); b -= b.mean(); c -= c.mean(); e -= e.mean()
k = .4
f = lambda p: .5 * p[0] ** 2 - .5 * k * p[1] ** 2 + .25 * p[1] ** 4
g = lambda p: np.array([p[0], -k * p[1] + p[1] ** 3])
gi = lambda p, i: g(p) + np.array([a[i] + c[i] * p[0] + e[i] * p[1], b[i] - c[i] * p[1] + e[i] * p[0]])
check_grad("f", f, g, rand_pts(20, -1.5, 1.5))
check("mean of component gradients = grad f", np.allclose(np.mean([gi(np.array([.2, .5]), i) for i in range(n)], 0), g([.2, .5])))
check_hess_at("saddle", f, [0, 0], [[1, 0], [0, -k]])


def neon2(gi_, x0, d, eta, T, r, sig, R):
    x = x0 + sig * R.normal(size=d)
    for t in range(T):
        i = R.integers(0, n)
        x = x - eta * (gi_(x, i) - gi_(x0, i))
        if np.linalg.norm(x - x0) >= r:
            v = (x - x0) / np.linalg.norm(x - x0)
            return v, t + 1
    return None, T


R = np.random.default_rng(1)
outs = [neon2(gi, np.zeros(2), 2, .1, 400, .3, 1e-3, R) for _ in range(300)]
vs = [o[0] for o in outs if o[0] is not None]
H0 = np.diag([1, -k])
check("2D: at the saddle (lambda_min = -0.4) NEON2 returns a direction almost always", len(vs) > 280, f"{len(vs)}/300")
check("2D: returned directions have v^T H v <= -delta/2 with delta = 0.4", np.mean([v @ H0 @ v <= -k / 2 for v in vs]) > .95)

# d = 20 quadratic finite sum: f_i(x) = 1/2 x^T (H + C_i) x + a_i^T x, H = diag(lambda_min, spread in [0.1, 1])
d, m = 20, 200
R1 = np.random.default_rng(2)
Cs = R1.normal(0, .15, (m, d, d)); Cs = (Cs + Cs.transpose(0, 2, 1)) / 2; Cs -= Cs.mean(0)
As = R1.normal(0, 1, (m, d)); As -= As.mean(0)
delta, eta, sig, rad = .2, .05, 1e-4, .05
T = int(np.ceil(1.4 * np.log(rad / sig) / (eta * delta)))
print(f"  d = {d}, delta = {delta}, eta = {eta}, T = {T}, r = {rad}, sigma = {sig}")
grid = np.round(np.linspace(-1.6, .6, 23), 3)
rows = []
for lr_ in grid:
    lam = lr_ * delta
    Hd = np.diag(np.concatenate([[lam], np.linspace(.1, 1, d - 1)]))
    gq = lambda x, i, Hd=Hd: (Hd + Cs[i]) @ x + As[i]
    ret, q = 0, []
    for s in range(150):
        v, _ = neon2(gq, np.zeros(d), d, eta, T, rad, sig, R1)
        if v is not None:
            ret += 1; q.append(v @ Hd @ v / delta)
    rows.append([float(lr_), ret / 150, (float(np.mean(q)) if q else None), (float(np.max(q)) if q else None), int(np.sum(np.array(q) > -.5)) if q else 0, ret])
    print(f"  lambda_min/delta = {lr_:+.2f}: P(return v) = {ret / 150:.2f}" + (f", mean v^T H v/delta = {np.mean(q):+.3f}, max {np.max(q):+.3f}" if q else ""))
check("lambda_min <= -delta: NEON2 returns v (>= 95%)", all(rw[1] >= .95 for rw in rows if rw[0] <= -1))
check("lambda_min >= -delta/2: NEON2 returns bottom (>= 95%)", all(rw[1] <= .05 for rw in rows if rw[0] >= -.5))
bad, tot = sum(rw[4] for rw in rows), sum(rw[5] for rw in rows)
check("returned v have v^T H v <= -delta/2 except with small probability (Theorem 1 holds w.p. >= 1 - p)", bad <= .01 * tot, f"{bad} of {tot} returns violate")
save_json("allenzhu2018neon2", "data.json", {"comp": [r(a, 5), r(b, 5), r(c, 5), r(e, 5)], "k": k, "rows": rows, "delta": delta, "eta": eta, "T": T, "d": d, "r": rad, "sigma": sig})
done()
