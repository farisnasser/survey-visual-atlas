"""Guide for survey v3, rewritten from the study notes (2026-11-23): every worked number the chapter texts quote and
every number the new visuals print, recomputed independently of the JavaScript."""
import math
import numpy as np
from common import check, done

print("guide3/notes")
# ---------- chapter 2 (II-A, II-B)
check("ch2: average of losses 1, 3, 2, 6 is 3", np.mean([1, 3, 2, 6]) == 3)
check("ch2: (uv-1)^2 is zero at (1,1) and (2,1/2); L(u,0) = 1", (1 * 1 - 1) ** 2 == 0 and (2 * .5 - 1) ** 2 == 0 and (0 - 1) ** 2 == 1)
Huv = np.array([[0, -2], [-2, 0]]); w, V = np.linalg.eigh(Huv)
check("ch2: Hessian of (uv-1)^2 at 0 has eigenvalues -2 along u=v and +2 along u=-v",
      np.allclose(w, [-2, 2]) and np.allclose(abs(V[:, 0]), [1 / np.sqrt(2)] * 2) and V[0, 1] * V[1, 1] < 0)
def gd_uv(a, T=600, eta=.05):
    p = 1.6 * np.array([math.cos(a), math.sin(a)])
    for _ in range(T):
        u, v = p; p = p - eta * np.array([2 * (u * v - 1) * v, 2 * (u * v - 1) * u])
    return p
ends = [gd_uv(math.radians(d)) for d in (20, 70, 200, 250, 130)]
check("ch2: GD from different starts reaches different zero-loss points of uv = 1", all(abs(p[0] * p[1] - 1) < 1e-4 for p in ends[:4])
      and np.std([p[0] for p in ends[:4]]) > .3, str([tuple(np.round(p, 3)) for p in ends]))
check("ch2: g=(3,4), Delta=(-.3,-.4): g.Delta = -2.5; |(3,4)| = 5", np.isclose(np.dot([3, 4], [-.3, -.4]), -2.5) and np.isclose(np.hypot(3, 4), 5))
check("ch2: with r = 0.5 the steepest step -r g/|g| is (-0.3, -0.4)", np.allclose(-.5 * np.array([3, 4]) / 5, [-.3, -.4]))
a, b = 1.3, -.7; H = np.array([[2, 3], [3, 5]]); d = np.array([a, b])
check("ch2: Delta^T H Delta = 2a^2 + 6ab + 5b^2 for H = [[2,3],[3,5]]", np.isclose(d @ H @ d, 2 * a * a + 6 * a * b + 5 * b * b))
# ---------- chapter 3 (II-C)
check("ch3: 5x^2 has gradient 10x, beta = 10, 2/beta = 0.2", np.isclose(2 / 10, .2))
check("ch3: eta <= 1/beta makes 1 - beta*eta/2 >= 1/2", 1 - 10 * .1 / 2 == .5)
a = 4.0
check("ch3: on a x^2/2 the multiplier 1 - eta a: <1 slow, 0 at 1/a, -1 at 2/a",
      np.isclose(1 - .5 / a * a, .5) and 1 - (1 / a) * a == 0 and 1 - (2 / a) * a == -1)
# ---------- chapter 4 (II-D, II-E)
check("ch4: 0.99^10 ~ 0.904 and 0.99^100 ~ 0.366", round(.99 ** 10, 3) == .904 and round(.99 ** 100, 3) == .366)
c, s_ = math.cos(math.pi / 4), math.sin(math.pi / 4); Q = np.array([[c, s_], [-s_, c]])
Hr = Q.T @ np.diag([100, 1]) @ Q
check("ch4: rotated Hessian is [[50.5,-49.5],[-49.5,50.5]] (or its mirror)", np.allclose(abs(Hr), [[50.5, 49.5], [49.5, 50.5]]))
Hr = np.array([[50.5, -49.5], [-49.5, 50.5]])
check("ch4: its inverse is [[.505,.495],[.495,.505]]", np.allclose(np.linalg.inv(Hr), [[.505, .495], [.495, .505]]))
t0 = np.array([1., 0.]); g0 = Hr @ t0; Lr = lambda t: .5 * t @ Hr @ t
check("ch4: L(1,0) = 25.25, g0 = (50.5,-49.5)", np.isclose(Lr(t0), 25.25) and np.allclose(g0, [50.5, -49.5]))
t_gd = t0 - .01 * g0; t_dg = t0 - g0 / 50.5; t_nt = t0 - np.linalg.solve(Hr, g0)
check("ch4: GD -> (0.495,0.495), L=0.245025", np.allclose(t_gd, [.495, .495]) and np.isclose(Lr(t_gd), .245025))
check("ch4: diagonal -> (0, 0.9802), L ~ 24.26", np.allclose(t_dg, [0, .9802], atol=1e-4) and round(Lr(t_dg), 2) == 24.26, f"{Lr(t_dg):.4f}")
check("ch4: Newton -> (0,0)", np.allclose(t_nt, 0))
check("ch4: powers 0.8 -> .64 -> .512, 1.2 -> 1.44 -> 1.728, -2 -> 4 -> -8 -> 16",
      np.allclose([.8 ** 2, .8 ** 3, 1.2 ** 2, 1.2 ** 3], [.64, .512, 1.44, 1.728]) and [(-2) ** k for k in range(1, 5)] == [-2, 4, -8, 16])
check("ch4: table multipliers A 0.5, B 1.5, C -1.5", np.allclose([1 - .1 * 5, 1 - .1 * -5, 1 - .5 * 5], [.5, 1.5, -1.5]))
K = np.diag(1 / np.sqrt([50.5, 50.5])) @ Hr @ np.diag(1 / np.sqrt([50.5, 50.5]))
check("ch4: diagonal metric at 45 deg: K eigenvalues 0.0198 and 1.98", np.allclose(np.linalg.eigvalsh(K), [.0198, 1.98], atol=5e-4))
# ---------- chapter 5 (II-F)
check("ch5: xi = (4.5,1.7) - (4,2) = (0.5,-0.3)", np.allclose(np.subtract([4.5, 1.7], [4, 2]), [.5, -.3]))
check("ch5: batch averages (2+8+10)/3 ~ 6.67, (4+6+8)/3 = 6", round(20 / 3, 2) == 6.67 and (4 + 6 + 8) / 3 == 6)
import itertools
xs = [2, 4, 6, 8, 10]; s2 = np.var(xs)
ok = s2 == 8
for B in (1, 2, 3, 4):
    means = [np.mean(c) for c in itertools.product(xs, repeat=B)]
    ok &= np.isclose(np.mean(means), 6) and np.isclose(np.var(means), 8 / B)
check("ch5: exact minibatch-mean distribution of 2..10 has mean 6, variance 8/B (B = 1..4)", ok)
check("ch5: sigma^2 = 100: B=10 -> 10, B=100 -> 1; sqrt(10) ~ 3.16", 100 / 10 == 10 and 100 / 100 == 1 and round(math.sqrt(10), 2) == 3.16)
# ---------- chapter 6 (II-G)
check("ch6: GD step -0.1 (4,2) = (-0.4,-0.2)", np.allclose(-.1 * np.array([4, 2]), [-.4, -.2]))
check("ch6: argmin 4D + 10D^2 is -0.2", np.isclose(-4 / 20, -.2))
check("ch6: M = diag(2,1): step (-0.2,-0.2)", np.allclose(-.1 * np.linalg.solve(np.diag([2, 1]), [4, 2]), [-.2, -.2]))
# ---------- chapter 7 (III-A..C)
check("ch7: w1 = 3 - 0.05*30 = 1.5; losses 45 -> 11.25", 3 - .05 * 30 == 1.5 and 5 * 9 == 45 and 5 * 1.5 ** 2 == 11.25)
check("ch7: heavy ball v1 = 5, v2 = 8; with g2 = -4, v2 = 0", .8 * 5 + 4 == 8 and .8 * 5 - 4 == 0)
def hb_stable(eta_lam, mu):
    r = np.roots([1, -(1 + mu - eta_lam), mu]); return max(abs(r)) < 1
ok = all(hb_stable(x, mu) == (0 < x < 2 * (1 + mu)) for mu in (0, .5, .9) for x in np.linspace(.01, 4, 77) if abs(x - 2 * (1 + mu)) > 1e-3)
check("ch7: heavy ball on a quadratic settles iff 0 < eta*lambda < 2(1 + mu)", ok)
y = 2 + .5 * (2 - 3)
check("ch7: Nesterov look-ahead 1.5, L'(2) = 4, L'(1.5) = 3, w = 1.2", y == 1.5 and 2 * 2 == 4 and 2 * y == 3 and np.isclose(y - .1 * 2 * y, 1.2))
# ---------- chapter 8 (III-D..H)
check("ch8: AdaGrad g = (10,1): s = (100,1), ratio (1,1)", np.allclose(np.array([10, 1]) / np.sqrt([100, 1]), [1, 1]))
check("ch8: RMSProp s1 = 10, s2 = 9.1", np.isclose(.1 * 100, 10) and np.isclose(.9 * 10 + .1 * 1, 9.1))
check("ch8: Adam m1 = 0.4, m_hat = 4", np.isclose(.1 * 4, .4) and np.isclose(.4 / .1, 4))
g = np.array([2., 8.]); m1 = .1 * g; v1 = .01 * g ** 2
check("ch8: Adam first step (2,8): m1 (0.2,0.8), v1 (0.04,0.64), ratio (1,1)",
      np.allclose(m1, [.2, .8]) and np.allclose(v1, [.04, .64]) and np.allclose((m1 / .1) / np.sqrt(v1 / .01), [1, 1]))
check("ch8: AMSGrad running max of 9,4,7 is 9,9,9", list(np.maximum.accumulate([9, 4, 7])) == [9, 9, 9])
check("ch8: AdamW decay factor 0.999 takes (10,1) to (9.99,0.999)", np.isclose(1 - .1 * .01, .999) and np.allclose(.999 * np.array([10, 1]), [9.99, .999]))
# ---------- chapter 9 (III-I..K)
Qa = np.array([[4, 8, 12], [2, 4, 6]]); r = Qa.sum(1); cc = Qa.sum(0)
check("ch9: Adafactor row sums (24,12), column sums (6,12,18), total 36, exact rank-one reconstruction",
      list(r) == [24, 12] and list(cc) == [6, 12, 18] and Qa.sum() == 36 and np.allclose(np.outer(r, cc) / r.sum(), Qa))
check("ch9: Lion sign step (-0.01,+0.01,-0.01)", np.allclose(-.01 * np.sign([8, -.2, 3]), [-.01, .01, -.01]))
# ---------- chapter 10 (IV-A..D)
check("ch10: 5w^2 at 4: loss 80, grad 40, GD to 3.6, Newton step -4", 5 * 16 == 80 and 10 * 4 == 40 and np.isclose(4 - .01 * 40, 3.6) and -40 / 10 == -4)
check("ch10: valley at (1,1): GD step (-1,-0.01), Newton (-1,-1)",
      np.allclose(-.01 * np.array([100, 1]), [-1, -.01]) and np.allclose(-np.array([100, 1]) / np.array([100, 1]), [-1, -1]))
check("ch10: e_{t+1} <= 2 e_t^2: 0.1 -> 0.02 -> 0.0008", np.isclose(2 * .1 ** 2, .02) and np.isclose(2 * .02 ** 2, .0008))
x = .2; gx = 4 * x ** 3 - 2 * x; hx = 12 * x ** 2 - 2; dx = -gx / hx
check("ch10: x^4 - x^2 at 0.2: g = -0.368, H = -1.52, Newton step -0.242, g*step +0.089",
      np.isclose(gx, -.368) and np.isclose(hx, -1.52) and round(dx, 3) == -.242 and round(gx * dx, 3) == .089)
check("ch10: damped lambda = 3: 1.48, step +0.249", np.isclose(hx + 3, 1.48) and round(-gx / (hx + 3), 3) == .249)
check("ch10: trust ratios 3/4 = 0.75 and 0.1/4 = 0.025; cubes 1 and 27", 3 / 4 == .75 and np.isclose(.1 / 4, .025) and 3 ** 3 == 27)
# ---------- chapter 11 (IV-E..G)
check("ch11: 2w^2 from 3 to 2: s = -1, y = -4, y/s = 4", (4 * 2 - 4 * 3) / (2 - 3) == 4)
check("ch11: dense d = 10,000: 1e8 values, 400 MB; L-BFGS k = 10: 200,000 values, 0.8 MB",
      10_000 ** 2 == 1e8 and 1e8 * 4 / 1e6 == 400 and 2 * 10 * 10_000 == 200_000 and 200_000 * 4 / 1e6 == .8)
check("ch11: H = diag(4,1), v = (2,3): Hv = (8,3)", np.allclose(np.diag([4, 1]) @ [2, 3], [8, 3]))
# ---------- chapter 12 (IV-H..J)
check("ch12: natural gradient F = diag(100,1), g = (2,2): F^-1 g = (0.02,2), step (-0.002,-0.2)",
      np.allclose(np.linalg.solve(np.diag([100, 1]), [2, 2]), [.02, 2]) and np.allclose(-.1 * np.array([.02, 2]), [-.002, -.2]))
check("ch12: A = 20 x 3 s = 60 s, B = 100 x 0.1 s = 10 s", 20 * 3 == 60 and np.isclose(100 * .1, 10))
# ---------- chapter 13 (V-A, V-B)
check("ch13: m = n = 100: 10,000 weights, 1e8 curvature entries, factors 2 x 10,000", 100 * 100 == 10_000 and 10_000 ** 2 == 1e8)
a_ = np.array([2, 1]); dl = np.array([3, 4])
check("ch13: aa^T = [[4,2],[2,1]], dd^T = [[9,12],[12,16]]", np.allclose(np.outer(a_, a_), [[4, 2], [2, 1]]) and np.allclose(np.outer(dl, dl), [[9, 12], [12, 16]]))
check("ch13: diag(2,3) kron diag(4,5) = diag(8,10,12,15)", np.allclose(np.kron(np.diag([2, 3]), np.diag([4, 5])), np.diag([8, 10, 12, 15])))
A_ = np.diag([4., 1.]); S_ = np.diag([9., 16.]); G_ = np.array([[6., 3.], [8., 4.]])
check("ch13: GA^-1 = [[1.5,3],[2,4]], S^-1 G A^-1 = [[1/6,1/3],[1/8,1/4]]",
      np.allclose(G_ @ np.linalg.inv(A_), [[1.5, 3], [2, 4]]) and np.allclose(np.linalg.inv(S_) @ G_ @ np.linalg.inv(A_), [[1 / 6, 1 / 3], [1 / 8, 1 / 4]]))
check("ch13: K-FAC identity (A kron S)^-1 = A^-1 kron S^-1, and kron applies as S^-1 G A^-1",
      np.allclose(np.linalg.inv(np.kron(A_, S_)), np.kron(np.linalg.inv(A_), np.linalg.inv(S_))) and
      np.allclose(np.linalg.solve(np.kron(A_, S_), G_.flatten(order="F")), (np.linalg.inv(S_) @ G_ @ np.linalg.inv(A_)).flatten(order="F")))
check("ch13: m = 4, n = 3 layer: factors 9 + 16 = 25 vs 12^2 = 144", 3 ** 2 + 4 ** 2 == 25 and 12 ** 2 == 144)
# ---------- chapter 14 (V-C)
check("ch14: 16^(1/4) = 2; diag(16,81)^(-1/4) = diag(1/2,1/3)", np.isclose(16 ** .25, 2) and np.allclose(np.array([16, 81]) ** -.25, [.5, 1 / 3]))
Gs = np.diag([2., 4.]); Ls = Gs @ Gs.T; Rs = Gs.T @ Gs; Lq = np.diag(np.diag(Ls) ** -.25)
check("ch14: G = diag(2,4): L = R = diag(4,16), L^-1/4 = diag(1/sqrt2,1/2), L^-1/4 G R^-1/4 = I",
      np.allclose(Ls, np.diag([4, 16])) and np.allclose(Lq, np.diag([1 / math.sqrt(2), .5])) and np.allclose(Lq @ Gs @ Lq, np.eye(2)))
check("ch14: one weight: v^-1/4 g v^-1/4 = g/sqrt(v)", np.isclose(9 ** -.25 * 3 * 9 ** -.25, 3 / 3))
# ---------- chapter 15 (V-D, V-E)
Hd = np.array([[10., 3.], [3., 2.]])
check("ch15: D^-1 g = (1,1) for D = diag(10,2), g = (10,2)", np.allclose(np.array([10, 2]) / np.array([10, 2]), [1, 1]))
rows = [(z, Hd @ z, np.array(z) * (Hd @ z)) for z in ([1, 1], [1, -1], [-1, 1], [-1, -1])]
check("ch15: Hutchinson table: Hz and z*Hz rows, average (10,2)",
      [tuple(r[1]) for r in rows] == [(13, 5), (7, 1), (-7, -1), (-13, -5)] and [tuple(r[2]) for r in rows] == [(13, 5), (7, -1), (7, -1), (13, 5)]
      and np.allclose(np.mean([r[2] for r in rows], 0), [10, 2]))
check("ch15: Apollo 5w^2 from 3 to 2: s = -1, y = -10, y/s = 10", (20 - 30) / (2 - 3) == 10)
check("ch15: rotated valley diagonal (50.5,50.5), eigenvalues 1 and 100", np.allclose(np.diag(Hr), 50.5) and np.allclose(np.linalg.eigvalsh(Hr), [1, 100]))
check("ch15: -g/H with g = 6: -1, -3, -60", np.allclose([-6 / 6, -6 / 2, -6 / .1], [-1, -3, -60]))
m_ = np.array([6., 6.]); h_ = np.array([2., .1]); ratio = m_ / np.maximum(h_, .01)
check("ch15: Sophia ratio (3,60), clipped (3,5), step (-0.3,-0.5); unclipped second move -6",
      np.allclose(ratio, [3, 60]) and np.allclose(np.clip(ratio, -5, 5), [3, 5]) and np.allclose(-.1 * np.clip(ratio, -5, 5), [-.3, -.5]) and np.isclose(-.1 * ratio[1], -6))
check("ch15: Sophia with h = -2 uses max(h, eps) = 0.01", max(-2, .01) == .01)
# ---------- chapter 16 (V-F..H)
U = np.array([[1, 1], [1, -1]]) / math.sqrt(2); Gm = np.array([[3., 1.], [1., 3.]])
check("ch16: SOAP: U^T G V = diag(4,2); U diag(1,2) V^T = [[1.5,-0.5],[-0.5,1.5]]",
      np.allclose(U.T @ Gm @ U, np.diag([4, 2])) and np.allclose(U @ np.diag([1, 2]) @ U.T, [[1.5, -.5], [-.5, 1.5]]))
check("ch16: Muon: GD -0.1 diag(4,1) = diag(-0.4,-0.1); polar factor of diag(4,1) is I",
      np.allclose(-.1 * np.diag([4, 1]), np.diag([-.4, -.1])) and np.allclose(np.linalg.svd(np.diag([4., 1.]))[0] @ np.linalg.svd(np.diag([4., 1.]))[2], np.eye(2)))
Gd = np.diag([4., 1.])
check("ch16: spectral budget 0.1: predicted changes -0.4 and -0.5; min = -eta * sum(sigma) = -0.5",
      np.isclose(np.sum(Gd * np.diag([-.1, 0])), -.4) and np.isclose(np.sum(Gd * np.diag([-.1, -.1])), -.5) and np.isclose(-.1 * np.linalg.svd(Gd)[1].sum(), -.5))
ns = lambda s: .5 * s * (3 - s * s)
check("ch16: classical cubic Newton-Schulz: 0.5 -> 0.6875, 0.9 -> 0.9855", np.isclose(ns(.5), .6875) and np.isclose(ns(.9), .9855))
check("ch16: (3,4)/5 = (0.6,0.8)", np.allclose(np.array([3, 4]) / 5, [.6, .8]))
# ---------- chapter 17 (VI)
check("ch17: 100 x 100: Adam 20,000, two factors 20,000, full block 1e8", 2 * 100 * 100 == 20_000 and 2 * 100 ** 2 == 20_000 and (100 * 100) ** 2 == 1e8)
m, n = 4096, 1024
check("ch17: 4096 x 1024: mn ~ 4.2M, Adam 8.4M, Adafactor 5,120, factors ~ 17.8M, full ~ 17.6T",
      round(m * n / 1e6, 1) == 4.2 and round(2 * m * n / 1e6, 1) == 8.4 and m + n == 5120 and round((m * m + n * n) / 1e6, 1) == 17.8 and round((m * n) ** 2 / 1e12, 1) == 17.6)
check("ch17: m^2 + n^2 - 2mn = (m - n)^2 >= 0", m * m + n * n - 2 * m * n == (m - n) ** 2)
check("ch17: students 10 x 20 = 200 s, 30 x 2 = 60 s; optimizers 100 x 0.5 = 50 s, 250 x 0.1 = 25 s",
      10 * 20 == 200 and 30 * 2 == 60 and 100 * .5 == 50 and np.isclose(250 * .1, 25))
check("ch17: C/T with C = 100: 10, 1, 0.1", [100 / T for T in (10, 100, 1000)] == [10, 1, .1])
done()
