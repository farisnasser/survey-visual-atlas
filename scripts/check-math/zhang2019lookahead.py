"""Zhang, Lucas, Hinton, Ba 2019: Lookahead optimizer, k steps forward, 1 step back.

Algorithm 1: fast weights theta take k inner steps from phi; slow weights phi <- phi + alpha (theta_k - phi).
Noisy quadratic (Sec. 3.1, Eq. 4): L^(x) = 1/2 (x - c)^T A (x - c), c ~ N(0, Sigma), A, Sigma diagonal.
Proposition 2 (per coordinate, r = 1 - gamma a):
    V_SGD = gamma^2 a^2 sigma^2 / (1 - r^2)
    V_LA  = alpha^2 (1 - r^{2k}) / (alpha^2 (1 - r^{2k}) + 2 alpha (1 - alpha)(1 - r^k)) * V_SGD   (slow weights)
"""
from common import *

print("zhang2019lookahead")


def v_sgd(g, a, s2):
    r = 1 - g * a
    return g * g * a * a * s2 / (1 - r * r)


def v_la(g, a, s2, k, al):
    r = 1 - g * a
    return al * al * (1 - r ** (2 * k)) / (al * al * (1 - r ** (2 * k)) + 2 * al * (1 - al) * (1 - r ** k)) * v_sgd(g, a, s2)


rng = np.random.default_rng(0)
for g, a, s2, k, al in [(.1, 1., 1., 5, .5), (.4, 2., .5, 10, .8), (.05, 4., 1., 3, .2)]:
    n = 200000; x = np.zeros(n)
    for _ in range(400):
        x = (1 - g * a) * x + g * a * rng.normal(0, np.sqrt(s2), n)
    mc_s = x.var()
    phi = np.zeros(n)
    for _ in range(int(400 / k) + 30):
        th = phi.copy()
        for _ in range(k):
            th = (1 - g * a) * th + g * a * rng.normal(0, np.sqrt(s2), n)
        phi = phi + al * (th - phi)
    mc_l = phi.var()
    check(f"Prop. 2 V_SGD (gamma {g}, a {a})", abs(mc_s - v_sgd(g, a, s2)) / v_sgd(g, a, s2) < .02, f"MC {mc_s:.4f} vs {v_sgd(g, a, s2):.4f}")
    check(f"Prop. 2 V_LA (k {k}, alpha {al})", abs(mc_l - v_la(g, a, s2, k, al)) / v_la(g, a, s2, k, al) < .02, f"MC {mc_l:.4f} vs {v_la(g, a, s2, k, al):.4f}")
# remark after Prop. 2: ratio < 1 for alpha in (0, 1), equal at alpha = 1
al = np.linspace(.01, .99, 99)
ok = all(v_la(g, a, 1, k, x) < v_sgd(g, a, 1) for g, a in [(.1, 1), (.5, 3), (.9, 2)] for k in [1, 5, 20] for x in al if 0 < g * a < 2)
check("V_LA < V_SGD for every alpha in (0, 1) (stable gamma)", ok)
check("alpha = 1 recovers SGD", abs(v_la(.1, 1, 1, 5, 1.) - v_sgd(.1, 1, 1)) < 1e-12)
# expected loss, Eq. (5), for the 3D tab: A = diag(1, 0.3), sigma^2 = (1, 1), gamma = 0.6, k = 5, alpha = 0.5
A2, S2, G3, K3, AL3 = np.array([1., .3]), np.array([1., 1.]), .6, 5, .5
Ls = .5 * np.sum(A2 * (v_sgd(G3, A2, S2) + S2)); Ll = .5 * np.sum(A2 * (v_la(G3, A2, S2, K3, AL3) + S2))
print(f"    3D tab steady-state expected loss: SGD {Ls:.4f}, Lookahead slow weights {Ll:.4f} (noise floor {.5 * np.sum(A2 * S2):.4f})")
check("3D tab: Lookahead slow weights have lower steady-state risk", Ll < Ls)
# excess risk (above the noise floor) that the page quotes
ex_s, ex_l = .5 * np.sum(A2 * v_sgd(G3, A2, S2)), .5 * np.sum(A2 * v_la(G3, A2, S2, K3, AL3))
check("3D tab: excess risk ratio", 0 < ex_l / ex_s < 1, f"SGD {ex_s:.4f}, LA {ex_l:.4f}, ratio {ex_l / ex_s:.3f}")
# Prop. 1: optimal alpha on a quadratic minimizes L(phi + alpha (theta_k - phi))
A = np.array([[2., .3], [.3, 1.]]); b = np.array([1., -1.]); xs = np.linalg.solve(A, b)
p0, pk = np.array([2., 1.]), np.array([.7, -.4])
astar = (p0 - xs) @ A @ (p0 - pk) / ((p0 - pk) @ A @ (p0 - pk))
Lq = lambda x: .5 * x @ A @ x - b @ x
grid = np.linspace(-1, 3, 40001); best = grid[np.argmin([Lq(p0 + t * (pk - p0)) for t in grid])]
check("Prop. 1 optimal alpha formula", abs(best - astar) < 1e-3, f"alpha* = {astar:.4f}")
# notes examples
check("notes: phi1 = 10 + 0.5 (8 - 10) = 9, phi2 = 9 + 0.5 (7 - 9) = 8", 10 + .5 * (8 - 10) == 9 and 9 + .5 * (7 - 9) == 8)
check("notes: EMA of independent endpoints has variance alpha sigma^2 / (2 - alpha) = 1/3 at alpha = 1/2", abs(.5 / 1.5 - 1 / 3) < 1e-12)
done()
