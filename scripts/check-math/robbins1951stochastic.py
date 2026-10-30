"""Robbins & Monro 1951: stochastic approximation x_{n+1} = x_n + a_n (alpha - y_n), E[y_n | x_n] = M(x_n).

Conditions: (4) bounded responses, (5) M(x) < alpha left of theta and > alpha right, (6) sum a_n^2 < inf,
(26) (implies (27) sum a_n = inf); a_n of "type 1/n" (c'/n <= a_n <= c''/n, Eq. 30) satisfies both.
Theorem 1/2: b_n = E(x_n - theta)^2 -> 0, hence x_n -> theta in probability.
1-D instance: M(x) = tanh(x), alpha = 0.5, theta = atanh(0.5); y_n = M(x_n) + U(-1, 1) (bounded).
2-D instance for the 3-D tab (the modern SGD reading): F(x, y) = log cosh x - x/2 + log cosh y + 0.3 y, gradient
(tanh x - 1/2, tanh y + 0.3) observed with independent U(-1, 1) noise; root (atanh 0.5, -atanh 0.3).
"""
from common import *

print("robbins1951stochastic")
alpha = .5; theta = np.arctanh(alpha)
check("M(x) = tanh x satisfies (5) around theta and M'(theta) > 0 (Theorem 2 conditions)", np.tanh(theta - .1) < alpha < np.tanh(theta + .1) and 1 - np.tanh(theta) ** 2 > 0)
check("type 1/n: a_n = 2/n has sum a_n^2 < inf and sum a_n = inf", np.isfinite(np.sum(4 / np.arange(1, 10 ** 6) ** 2)) and np.sum(2 / np.arange(1, 10 ** 6)) > 25)


def run1d(sched, N, seed, x1=-2.0):
    R = np.random.default_rng(seed); x = x1; out = np.empty(N)
    for n in range(1, N + 1):
        out[n - 1] = x
        y = np.tanh(x) + R.uniform(-1, 1)
        x = x + sched(n) * (alpha - y)
    return out


S = {"2/n": lambda n: 2 / n, "0.2": lambda n: .2, "2/n^2": lambda n: 2 / n ** 2}
N = 20000
B = {k: np.mean([(run1d(f, N, s) - theta) ** 2 for s in range(200)], 0) for k, f in S.items()}
for k in S:
    print(f"  a_n = {k:6s}: b_n = E(x_n - theta)^2 at n = 100, 1000, 20000: {B[k][99]:.4f}, {B[k][999]:.5f}, {B[k][-1]:.6f}")
check("a_n = 2/n (type 1/n): b_n -> 0, roughly like 1/n", B["2/n"][-1] < 1e-3 and B["2/n"][-1] < .2 * B["2/n"][1999])
check("constant a_n = 0.2 (sum a_n^2 = inf): b_n stalls at a noise floor", B["0.2"][-1] > 5 * B["2/n"][-1] and abs(B["0.2"][-1] - B["0.2"][9999]) < .5 * B["0.2"][-1])
check("a_n = 2/n^2 (sum a_n < inf, violates (27)): b_n stays away from 0", B["2/n^2"][-1] > .5)
# 2-D for the 3-D tab
g2 = lambda p: np.array([np.tanh(p[0]) - .5, np.tanh(p[1]) + .3])
F2 = lambda p: np.log(np.cosh(p[0])) - p[0] / 2 + np.log(np.cosh(p[1])) + .3 * p[1]
check_grad("F(x, y)", F2, g2, rand_pts(20, -2, 2))
root = np.array([np.arctanh(.5), -np.arctanh(.3)])
check("root of the 2-D gradient", np.allclose(g2(root), 0))
paths = []
for s in range(3):
    R = np.random.default_rng(100 + s)
    paths.append(sig(run1d(S["2/n"], 400, 100 + s), 4))
save_json("robbins1951stochastic", "data.json", {"theta": round(float(theta), 6), "root": r(root, 6), "paths": paths,
                                               "b": {k: sig(v[np.unique(np.round(np.logspace(0, np.log10(N - 1), 120)).astype(int))], 4) for k, v in B.items()},
                                               "bn": np.unique(np.round(np.logspace(0, np.log10(N - 1), 120)).astype(int)).tolist()})
done()
