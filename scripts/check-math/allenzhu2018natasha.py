"""Allen-Zhu 2018 (Natasha 2): swing by saddle points. Sec. 4.2 reduction (after Carmon et al.), with stochastic pieces.

Finite sum f = mean f_i, f_i(x, y) = (x^2 - 1)^2/4 + y^2/2 + a_i x + b_i y + c_i (x^2 - y^2)/2 (zero-mean a, b, c);
saddle (0, 0), minima (+-1, 0).  Round k at y_k:
  Oja's algorithm with stochastic Hessian-vector products H_i w (H_i = Hessian of f_i at y_k) estimates the most negative
  eigenvector v; if v^T H v <= -delta/2: move y_k +- (delta/L2) v  (negative-curvature step);
  else: minibatch SGD on F_k(x) = f(x) + L max(0, |x - y_k| - delta/L2)^2 (safe zone of radius delta/L2).
Simplification: the paper solves F_k with Natasha1.5 (SVRG-style); here plain minibatch SGD.
"""
from common import *

print("allenzhu2018natasha")
n = 200
R0 = np.random.default_rng(81)
a = R0.normal(0, .8, n); b = R0.normal(0, .8, n); c = R0.normal(0, .5, n)
a -= a.mean(); b -= b.mean(); c -= c.mean()
f = lambda p: .25 * (p[0] ** 2 - 1) ** 2 + .5 * p[1] ** 2
g = lambda p: np.array([p[0] ** 3 - p[0], p[1]])
gi = lambda p, I: np.stack([p[0] ** 3 - p[0] + a[I] + c[I] * p[0], p[1] + b[I] - c[I] * p[1]], -1)
Hi = lambda p, i: np.diag([3 * p[0] ** 2 - 1 + c[i], 1 - c[i]])
fi = lambda p, i: f(p) + a[i] * p[0] + b[i] * p[1] + c[i] * (p[0] ** 2 - p[1] ** 2) / 2
check_grad("f_3", lambda p: fi(p, 3), lambda p: gi(p, np.array([3]))[0], rand_pts(20, -1.6, 1.6))
check_hess_at("f_3 at (0.4, -0.2)", lambda p: fi(p, 3), np.array([.4, -.2]), Hi([.4, -.2], 3))
check("mean component Hessian = Hessian of f", np.allclose(np.mean([Hi([.3, .1], i) for i in range(n)], 0), np.diag([3 * .09 - 1, 1])))
B = 1.6
L = float(3 * B * B - 1 + np.max(np.abs(c)))      # gradient Lipschitz bound on the box (components)
L2 = 6 * B                                          # Hessian Lipschitz: d/dx (3x^2 - 1) = 6x
V = float(np.mean(np.sum((gi(np.zeros(2), np.arange(n)) - g(np.zeros(2))) ** 2, 1)))
print(f"  L = {L:.2f}, L2 = {L2:.2f}, variance at 0: {V:.3f}")
delta, EPS = .5, .05
rad = delta / L2


def oja(p, R, m=30, lr=.08):
    w = R.normal(size=2); w /= np.linalg.norm(w)
    for _ in range(m):
        i = R.integers(n); w = w - lr * Hi(p, i) @ w; w /= np.linalg.norm(w)
    return w


R = np.random.default_rng(1)
p = np.array([.3, .4])
ws = [oja(p, R) for _ in range(200)]
Hf = np.diag([3 * .09 - 1, 1])
check("Oja with 30 stochastic HVPs finds a direction with v^T H v <= -delta/2 (w.p. >= 2/3)", np.mean([w @ Hf @ w <= -delta / 2 for w in ws]) >= 2 / 3,
      f"success {np.mean([w @ Hf @ w <= -delta / 2 for w in ws]):.0%}, lambda_min = {eig(Hf)[0]:.2f}")


def natasha(x0, seed, rounds=400, inner=12, lr=.04, S=4):
    R = np.random.default_rng(seed); y = np.array(x0, float); T = 0; nc = 0; path = [y.copy()]
    for k in range(rounds):
        w = oja(y, R); T += 30
        Hy = np.mean([Hi(y, i) for i in range(n)], 0)
        if w @ Hy @ w <= -delta / 2:   # the paper checks the estimate with more samples; we use the exact quadratic form here
            s = min((y + rad * w, y - rad * w), key=f); y = s; nc += 1
        else:
            x = y.copy()
            for _ in range(inner):
                I = R.integers(0, n, S); d = x - y; nd = np.linalg.norm(d)
                pen = 2 * L * max(0, nd - rad) * d / nd if nd > 0 else 0
                x = x - lr * (gi(x, I).mean(0) + pen); T += S
            y = x
        path.append(y.copy())
        Hy = np.diag([3 * y[0] ** 2 - 1, 1])
        if np.linalg.norm(g(y)) <= EPS and eig(Hy)[0] >= -delta:
            return y, k + 1, T, nc, np.array(path)
    return y, rounds, T, nc, np.array(path)


x0 = np.array([0., 1.3])   # on the stable manifold of the saddle
res = [natasha(x0, s) for s in range(20)]
ok = [np.linalg.norm(g(r_[0])) <= EPS and eig(np.diag([3 * r_[0][0] ** 2 - 1, 1]))[0] >= -delta for r_ in res]
check("the loop ends at an (eps, delta)-approximate local minimum in every seeded run", all(ok), f"rounds median {np.median([r_[1] for r_ in res]):.0f}, NC steps median {np.median([r_[3] for r_ in res]):.0f}")
check("it never comes close to the saddle (min distance > 0.03)", min(np.min(np.linalg.norm(r_[4], axis=1)) for r_ in res) > .03,
      f"min distance {min(np.min(np.linalg.norm(r_[4], axis=1)) for r_ in res):.3f}")


gdp = gd(g, x0, .1, 300)
check("full-gradient GD from the same start converges to the saddle (stable manifold x = 0)", np.linalg.norm(gdp[-1]) < 1e-6)
nr = [int(np.sum(np.linalg.norm(gdp, axis=1) < .15)) for _ in [0]][0]
print(f"  GD iterates within 0.15 of the saddle: {nr} of 300")
# Theorem 2 / Remark 4.4 orders: T = 1/delta^5 + 1/(delta eps^3) + eps^-3.25; equals eps^-3.25 order when delta = eps^(1/4)
e = 1e-3; d_ = e ** .25
check("with delta = eps^(1/4): 1/delta^5 + 1/(delta eps^3) = eps^-1.25 + eps^-3.25", abs((d_ ** -5 + 1 / (d_ * e ** 3)) / (e ** -1.25 + e ** -3.25) - 1) < 1e-9)
save_json("allenzhu2018natasha", "data.json", {"a": r(a, 5), "b": r(b, 5), "c": r(c, 5), "L": round(L, 4), "L2": round(L2, 4), "delta": delta, "eps": EPS})
done()
