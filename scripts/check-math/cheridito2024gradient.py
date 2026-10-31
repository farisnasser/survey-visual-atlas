"""Cheridito, Jentzen, Rossmannek 2024 (JOTA; arXiv 2208.02083): shallow ReLU nets N(x) = c + sum_j v_j max(w_j x + b_j, 0) on
[alpha, beta], affine target f, loss L = int (N - f)^2 dx, plain GD.
Prop. 4.5 (iv): every critical point that is not a global minimum has L = f'(alpha)^2 (beta - alpha)^3 / (12 (n+1)^4), n in {0, 2, 4, ...},
n <= N, and the network is a sawtooth around f. (ii): it is a local min iff c = f(mid) and every neuron is inactive or semi-inactive with
f'(alpha) v_j w_j < 0 (dead neurons: spurious local minima at level n = 0). Theorem 1.2 (N even): for a.e. step and a.e. start whose GD
limit has loss < f'(alpha)^2 (beta - alpha)^3 / (12 (N-1)^4), the limit loss is 0.

Here alpha = 0, beta = 1, f(x) = x; integrals by a 4000-point midpoint rule (exact up to O(h^2) at the kinks).
"""
from common import *

print("cheridito2024gradient")
XQ = (np.arange(4200) + .5) / 4200; WQ = 1 / 4200          # 4200 cells: kinks at multiples of 1/3 and 1/5 fall on cell edges
def net(th, N, x=XQ):
    w, b, v, c = th[:N], th[N:2 * N], th[2 * N:3 * N], th[3 * N]
    return c + np.maximum(np.outer(x, w) + b, 0) @ v
def loss(th, N): return np.sum((net(th, N) - XQ) ** 2) * WQ
def grad(th, N):
    w, b, v = th[:N], th[N:2 * N], th[2 * N:3 * N]; Z = np.outer(XQ, w) + b; A = np.maximum(Z, 0); act = (Z > 0) * 1.
    r = 2 * (A @ v + th[3 * N] - XQ) * WQ
    return np.concatenate([(r[:, None] * act * XQ[:, None]).sum(0) * v, (r[:, None] * act).sum(0) * v, A.T @ r, [r.sum()]])
th = np.random.default_rng(0).normal(size=13)
check_grad("integral loss, N = 4 (away from kinks)", lambda t: loss(t, 4), lambda t: grad(t, 4), [th])
Ln = lambda n: 1 / (12 * (n + 1) ** 4)
# the sawtooth networks of Prop. 4.5 (iv) attain exactly these levels
def sawtooth(n, x):
    i = np.minimum(np.floor(x * (n + 1)), n)
    return x - (-1) ** i / (n + 1) * (x - (i + .5) / (n + 1))
for n in [0, 2, 4]:
    val = np.sum((sawtooth(n, XQ) - XQ) ** 2) * WQ
    check(f"sawtooth with n = {n} kinks has loss f'^2 (b-a)^3 / (12 (n+1)^4) = {Ln(n):.3e}", abs(val - Ln(n)) < 1e-6 * Ln(n) + 1e-9)
# n = 2 sawtooth as a ReLU net with the sign pattern of Prop. 4.5 (iii)(b): the neuron at kink i has sign(w) = (-1)^(i+1) and
# v w = 2 f'(alpha) / (n + 1); c = f(mid) = 1/2; the other two neurons are dead (w = -1, b = -1, v = 0).
def saw_net(n, N):
    w = [1., -1.] + [-1.] * (N - 2); b = [-1 / 3, 2 / 3] + [-1.] * (N - 2); v = [2 / 3, -2 / 3] + [0.] * (N - 2)
    return np.array(w + b + v + [.5])
t2 = saw_net(2, 4); err = np.max(np.abs(net(t2, 4) - sawtooth(2, XQ)))
check("the n = 2 sawtooth is realised by a ReLU net with N = 4 neurons (one dead)", err < 1e-12)
check("and it is a critical point of L (gradient ~ 0)", np.linalg.norm(grad(t2, 4)) < 1e-6, f"|grad| = {np.linalg.norm(grad(t2, 4)):.1e}")
Hs = fd_hess(lambda t: loss(t, 4), t2, h=1e-4)
check("and a saddle: the Hessian has a negative eigenvalue", eig(Hs)[0] < -1e-4, f"lambda_min = {eig(Hs)[0]:.4f}")
# slice for the 3D tab: c = 1/2, neuron 1 with w = -1 (kink at x = b), neuron 2 dead; L(b, v)
def slice_L(b1, v1):
    return np.sum((.5 + v1 * np.maximum(b1 - XQ, 0) - XQ) ** 2) * WQ
check("slice: for b <= 0 the neuron is off and L = 1/12 for every v (a flat set of critical points)", all(abs(slice_L(b1, v1) - 1 / 12) < 1e-7 for b1 in [-.4, -.1, 0.] for v1 in [-2, 0, 2]))
check("slice: just inside b > 0 the loss falls if v < 0 and rises if v > 0 (saddle vs spurious local min, Prop. 4.5 ii-iii)", slice_L(.05, -1) < 1 / 12 < slice_L(.05, 1))
# closed form of the slice used on the page: int_0^min(b,1) (A - B x)^2 + int_min(b,1)^1 (1/2 - x)^2, A = 1/2 + v b, B = 1 + v
def slice_cf(b1, v1):
    if b1 <= 0: return 1 / 12
    m = min(b1, 1.); A = .5 + v1 * b1; B = 1 + v1
    return A * A * m - A * B * m * m + B * B * m ** 3 / 3 + (.125 - (m - .5) ** 3) / 3
check("slice closed form matches the quadrature", max(abs(slice_cf(b1, v1) - slice_L(b1, v1)) for b1 in np.linspace(-.3, 1.2, 16) for v1 in np.linspace(-2, 2, 9)) < 1e-7)
# GD on the full network, N = 4 (even), from random starts; final losses vs the critical levels and the Theorem 1.2 threshold
N = 4; thr = 1 / (12 * (N - 1) ** 4)
XB = (np.arange(1000) + .5) / 1000; WB = 1 / 1000          # coarser rule for the batched training runs
def bgrad(T):
    w, b, v, c = T[:, :N], T[:, N:2 * N], T[:, 2 * N:3 * N], T[:, 3 * N]
    Z = XB[None, :, None] * w[:, None, :] + b[:, None, :]; A = np.maximum(Z, 0); act = (Z > 0) * 1.
    r = 2 * (np.einsum("snj,sj->sn", A, v) + c[:, None] - XB[None, :]) * WB
    gw = np.einsum("sn,snj->sj", r, act * XB[None, :, None]) * v; gb = np.einsum("sn,snj->sj", r, act) * v
    return np.concatenate([gw, gb, np.einsum("sn,snj->sj", r, A), r.sum(1, keepdims=True)], 1)
S = 40; R = np.random.default_rng(0)
T = np.concatenate([R.normal(size=(S, N)), R.uniform(-1, 1, (S, N)), R.normal(size=(S, N)) * .5, R.normal(size=(S, 1)) * .5], 1)
T[:, 2 * N:] *= .02                                            # small output weights and bias: start near the n = 0 saddle level
marks = sorted(set(np.unique(np.round(np.logspace(0, np.log10(20000), 45)).astype(int) - 1).tolist())); curves = [[] for _ in range(S)]
for k in range(20000):
    if k in marks:
        for sd in range(S): curves[sd].append([k + 1, loss(T[sd], N)])
    T = T - .2 * bgrad(T)
rows = [[sd, loss(T[sd], N), np.linalg.norm(grad(T[sd], N))] for sd in range(S)]
rows = np.array(rows)
below = rows[rows[:, 1] < thr]; above = rows[rows[:, 1] >= thr]
print(f"  N = 4: threshold f'^2/(12 (N-1)^4) = {thr:.3e}; levels L_0 = {Ln(0):.3e}, L_2 = {Ln(2):.3e}, L_4 = {Ln(4):.3e}")
print(f"  40 GD runs (step 0.2, 20000 steps): final losses sorted {np.round(np.sort(rows[:, 1]), 6).tolist()}")
print(f"  below threshold: {len(below)}, max final loss among them {below[:, 1].max() if len(below) else 0:.2e}; above: {len(above)}")
check("every run that ends below the threshold is (numerically) at loss 0 (Theorem 1.2)", len(below) > 0 and below[:, 1].max() < 1e-5)
lv = [Ln(n) for n in (0, 2, 4)]
near = [min(abs(np.log(q / l)) for l in lv) for q in above[:, 1]]
check("runs above the threshold sit at one of the critical levels L_0, L_2, L_4 (within 5%) or are still moving", len(above) == 0 or np.mean(np.array(near) < .05) >= .5, f"{np.round(near, 3).tolist()}")
save_json("cheridito2024gradient", "data.json", {"thr": thr, "levels": lv, "final": [float(f"{q:.4g}") for q in rows[:, 1]], "curves": [[[a, float(f"{b:.4g}")] for a, b in cv] for cv in curves]})

# in-page notes (papers/cheridito2024gradient/notes.md): primer idea 4, one ReLU neuron, bias only, x ~ U[0,1], target x
gx, gw = np.polynomial.legendre.leggauss(8)        # exact for the piecewise polynomial integrand
def Lb(b):
    tot = 0.
    for lo, hi in [(0., min(max(-b, 0.), 1.)), (min(max(-b, 0.), 1.), 1.)]:
        if hi > lo:
            x = (hi - lo) / 2 * gx + (hi + lo) / 2
            tot += (hi - lo) / 2 * np.sum(gw * (np.maximum(x + b, 0) - x) ** 2)
    return tot
Lpw = lambda b: b * b if b >= 0 else (b * b + 2 * b ** 3 / 3 if b >= -1 else 1 / 3)
bs = np.linspace(-2, 1, 61)
check("primer: L(b) = b^2 (b>=0), b^2 + 2b^3/3 (-1<=b<=0), 1/3 (b<=-1) matches the integral", max(abs(Lb(b) - Lpw(b)) for b in bs) < 1e-10)
d1 = lambda b, h=1e-4: (Lpw(b + h) - Lpw(b - h)) / (2 * h)
d2r = lambda b, h=1e-3: (Lpw(b + 2 * h) - 2 * Lpw(b + h) + Lpw(b)) / h ** 2
d2l = lambda b, h=1e-3: (Lpw(b) - 2 * Lpw(b - h) + Lpw(b - 2 * h)) / h ** 2
print(f"  primer: at b=0: L'={d1(0):.2e}, L'' right {d2r(0):.3f} left {d2l(0):.3f}; at b=-1: L'={d1(-1):.2e}, L'' right {d2r(-1):.3f} left {d2l(-1):.3f}")
check("primer: at b = 0, L = L' = 0 and L'' = 2 from both sides (kink smoothed to second order)", Lpw(0) == 0 and abs(d1(0)) < 1e-6 and abs(d2r(0) - 2) < 1e-2 and abs(d2l(0) - 2) < 1e-2)
d1r, d1l = (Lpw(-1 + 1e-7) - Lpw(-1)) / 1e-7, (Lpw(-1) - Lpw(-1 - 1e-7)) / 1e-7
check("primer: at b = -1, L' = 0 from both sides but L'' = -2 from the right and 0 from the left (C^1, not C^2)", abs(d1r) < 1e-5 and abs(d1l) < 1e-8 and abs(d2r(-1) + 2) < 1e-2 and abs(d2l(-1)) < 1e-9)
check("primer: max(x,0) maps (-inf,0] to {0} (preimage of a null set has positive length)", all(max(x, 0) == 0 for x in np.linspace(-5, 0, 11)))
done()
