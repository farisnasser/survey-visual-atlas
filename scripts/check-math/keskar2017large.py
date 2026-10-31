"""Keskar, Mudigere, Nocedal, Smelyanskiy, Tang 2017 (ICLR; arXiv 1609.04836): large-batch (LB) training converges to sharp
minimizers and generalizes worse than small-batch (SB); sharpness metric (4): phi = (max_{y in C_eps} f(x + y) - f(x)) / (1 + f(x)) * 100,
C_eps = {z : |z_i| <= eps (|x_i| + 1)}. Their explanation (Sec. 4): SB gradient noise pushes iterates out of sharp basins.

Toy (a model of that explanation, not a network): finite sum f(w) = (1/n) sum_i [F(w) + v_i . w], n = 64 fixed tilts v_i with
sum v_i = 0 (so f = F exactly), F = the two-basin function of the flatness page (flat minimum near (-1, 0), sharp near (1, 0)).
Minibatch SGD with batch B (indices drawn with replacement): gradient noise = mean of B tilts, std ~ sigma / sqrt(B).
"""
from common import *

print("keskar2017large")
SF, SS, SIGV, N = .55, .12, 5., 64
def F(P):
    x, y = P[..., 0], P[..., 1]
    return .02 * (x * x + y * y) - np.exp(-((x + 1) ** 2 + y ** 2) / (2 * SF ** 2)) - np.exp(-((x - 1) ** 2 + y ** 2) / (2 * SS ** 2))
def dF(P):
    x, y = P[..., 0], P[..., 1]
    a = np.exp(-((x + 1) ** 2 + y ** 2) / (2 * SF ** 2)) / SF ** 2; b = np.exp(-((x - 1) ** 2 + y ** 2) / (2 * SS ** 2)) / SS ** 2
    return np.stack([.04 * x + a * (x + 1) + b * (x - 1), .04 * y + a * y + b * y], -1)
V = np.random.default_rng(0).normal(0, SIGV, (N, 2)); V -= V.mean(0)
f = lambda w: float(F(np.asarray(w, float)))          # the tilts average to zero
g = lambda w: dF(np.asarray(w, float))
check("the tilts sum to zero, so the full-batch loss is F itself", np.max(np.abs(V.sum(0))) < 1e-12)
check_grad("finite-sum f", f, g, rand_pts(20, -2, 2))
def newton(p):
    p = np.array(p, float)
    for _ in range(60): p = p - np.linalg.solve(fd_hess(f, p), g(p))
    return p
mF, mS = newton([-1, 0]), newton([1, 0])
print(f"  full-batch minima: flat {mF.round(4)} f = {f(mF):.4f}, sharp {mS.round(4)} f = {f(mS):.4f}; Hessian eigs {eig(fd_hess(f, mF)).round(2)} vs {eig(fd_hess(f, mS)).round(2)}")
check("both minima survive the averaging, the right one much sharper", eig(fd_hess(f, mS))[0] > 8 * eig(fd_hess(f, mF))[1] and np.linalg.norm(g(mS)) < 1e-9)
def sharp_metric(x, eps=1e-3, n=4000, seed=0):    # metric (4) with A = I by random search over the box corners/interior
    R = np.random.default_rng(seed); box = eps * (np.abs(x) + 1)
    Y = R.uniform(-1, 1, (n, 2)) * box; Y = np.vstack([Y, np.array([[a, b] for a in (-1, 1) for b in (-1, 1)]) * box])
    return (max(f(x + y) for y in Y) - f(x)) / (1 + f(x)) * 100
phF, phS = sharp_metric(mF), sharp_metric(mS)
print(f"  sharpness metric (4), eps = 1e-3: flat {phF:.4f}, sharp {phS:.4f}")
check("metric (4) is larger at the sharp minimizer", phS > 5 * phF)
ETA, T = .02, 20000
def run(B, starts, seed):
    R = np.random.default_rng(seed); W = starts.copy()
    for t in range(T):
        idx = R.integers(0, N, (len(W), B)); W = W - ETA * (dF(W) + V[idx].mean(1))
    return W
R0 = np.random.default_rng(1); starts = np.stack([R0.uniform(-2, 2, 400), R0.uniform(-1.2, 1.2, 400)], 1)
rows = []
for B in [1, 2, 3, 4, 6, 8, 16, 64]:
    W = run(B, starts, 10 + B); inF = np.mean(np.linalg.norm(W - mF, axis=1) < .4); inS = np.mean(np.linalg.norm(W - mS, axis=1) < .2)
    rows.append([B, float(inF), float(inS)]); print(f"  batch B = {B:2d}: end in the flat basin {inF:.3f}, in the sharp basin {inS:.3f}")
sh = {q[0]: q[2] for q in rows}
check("threshold in the batch size: B <= 2 ends in the sharp basin < 5% of the time, B >= 4 more than 15%", sh[1] < .05 and sh[2] < .05 and all(sh[b] > .15 for b in (4, 6, 8, 16, 64)))
check("with B = 1 the noise is so large that most runs have not settled in either basin", rows[0][1] + rows[0][2] < .5)
save_json("keskar2017large", "data.json", {"V": r(V, 5), "mF": r(mF, 5), "mS": r(mS, 5), "rows": rows, "phi": [phF, phS]})
# in-page notes (papers/keskar2017large/notes.md): Sec. 2 bound max_{|u|<=r} u^T H u / 2 = r^2 lambda_max / 2, teaching example
Rk = np.random.default_rng(1)
A = Rk.normal(size=(5, 5)); Hq = A @ A.T; rq = .3
U = Rk.normal(size=(200000, 5)); U = rq * U / np.linalg.norm(U, axis=1, keepdims=True)
check("notes: max over the r-ball of u^T H u / 2 is r^2 lambda_max / 2 (sampled max <= bound, eigenvector attains it)",
      np.max(.5 * np.einsum('ij,jk,ik->i', U, Hq, U)) <= .5 * rq ** 2 * eig(Hq)[-1] + 1e-12
      and np.isclose(.5 * rq ** 2 * eig(Hq)[-1], .5 * (rq * np.linalg.eigh(Hq)[1][:, -1]) @ Hq @ (rq * np.linalg.eigh(Hq)[1][:, -1])))
check("notes: L1 = x^2 + 100 y^2 rises 100x faster than L2 = x^2 + y^2 along y", np.isclose((0 + 100 * .01 ** 2) / (0 + .01 ** 2), 100))
# primer: steps per epoch, 1/sqrt(b) noise, shifted test loss, basin width vs noise kick
check("primer: n = 50,000 -> 196 steps (b = 256), 10 steps (b = 5,000 = 10%)", int(np.ceil(50000 / 256)) == 196 and 50000 // 5000 == 10 and 5000 == .1 * 50000)
sds = {b: float(np.std(Rk.choice([-1., 1.], size=(20000, b)).mean(1))) for b in [1, 16, 256]}
check("primer: per-example sd 1 -> minibatch sd 1, 0.25, 0.0625 (sampled)", all(np.isclose(sds[b], 1 / np.sqrt(b), rtol=.03) for b in sds), str({b: round(v, 4) for b, v in sds.items()}))
check("primer: shift 0.1 costs 0.01 (theta^2) vs 1 (100 theta^2)", np.isclose(.1 ** 2, .01) and np.isclose(100 * .1 ** 2, 1))
w = lambda lam, d=.01: np.sqrt(2 * d / lam); kick = lambda b, eta=.1, sg=1.: eta * sg / np.sqrt(b)
check("primer: half-widths 0.014 (lambda 100) / 0.14 (lambda 1); kicks 0.1 (b = 1), 0.006 (b = 256)",
      np.isclose(w(100), .014, atol=5e-4) and np.isclose(w(1), .14, atol=5e-3) and np.isclose(kick(1), .1) and np.isclose(kick(256), .006, atol=5e-4)
      and w(100) < kick(1) < w(1) and kick(256) < w(100), f"{w(100):.4f}, {w(1):.4f}, {kick(256):.5f}")
done()
