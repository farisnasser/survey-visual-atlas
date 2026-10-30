"""Xu & Du 2023: over-parameterization exponentially slows down GD for learning a single ReLU neuron.

L(w) = E_x 1/2 (sum_j relu(w_j . x) - relu(v . x))^2, x ~ N(0, I_d)  (exact via k(u, v) = (|u||v| sin phi + (pi - phi) u.v)/(2 pi)).
Gradient: Eq. (3) (Safran & Shamir 2018). Fig. 1 setting: sigma = 0.1, eta = 0.05, |v| = 1; init w_i ~ N(0, sigma^2 I).
Theorem 1: n >= 2 -> L(w(T)) <= poly * T^-3 for GD; Theorem 2: gradient flow L(w(t)) >= (G1 t + G2)^-3; n = 1: exp(-Omega(T)).
Slice for the 3-D tab (d = 2, v = e1): w1 = (a, b), w2 = (1 - a, -b): the students always sum to v and the loss comes only from
their angle; near b = 0 it grows like |b|^3, which makes GD on the slice algebraically slow (b ~ 1/t, loss ~ t^-3).
"""
from common import *

print("xu2023over")


def k(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    c = np.clip(u @ v / (nu * nv), -1, 1); ph = np.arccos(c)
    return (nu * nv * np.sin(ph) + (np.pi - ph) * (u @ v)) / (2 * np.pi)


def loss(W, v):
    return .5 * (sum(k(a, b) for a in W for b in W) - 2 * sum(k(a, v) for a in W) + k(v, v))


def grad(W, v):   # Eq. (3)
    n = len(W); G = np.zeros_like(W); nv = np.linalg.norm(v)
    ang = lambda a, b: np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    for i in range(n):
        th_i = ang(W[i], v)
        s = sum(np.linalg.norm(W[j]) * np.sin(ang(W[i], W[j])) for j in range(n) if j != i) - nv * np.sin(th_i)
        t = sum(ang(W[i], W[j]) * W[j] for j in range(n) if j != i)
        G[i] = .5 * (W.sum(0) - v) + (s * W[i] / np.linalg.norm(W[i]) - t + th_i * v) / (2 * np.pi)   # w-bar_i = w_i/|w_i|
    return G


d = 10
v = np.zeros(d); v[0] = 1
R = np.random.default_rng(92)
Wt = R.normal(0, 1, (3, d))
check_grad("Eq. (3) gradient (n = 3, d = 10)", lambda w: loss(w.reshape(3, d), v), lambda w: grad(w.reshape(3, d), v).ravel(), [Wt.ravel(), .3 * Wt.ravel()])
xs = R.normal(size=(400000, d)); relu = lambda z: np.maximum(z, 0)
check("closed-form loss matches Monte Carlo", abs(loss(Wt, v) - .5 * np.mean((relu(xs @ Wt.T).sum(1) - relu(xs @ v)) ** 2)) < 2e-2 * loss(Wt, v))
sigma, eta, T = .1, .05, 60000
curves = {}
for n in (1, 2, 3, 5):
    W = R.normal(0, sigma, (n, d)); L = []
    for t in range(T + 1):
        if t % 50 == 0 or t < 50:
            L.append((t, loss(W, v)))
        if t < T:
            W = W - eta * grad(W, v)
        if L[-1][1] < 1e-28:
            break
    curves[n] = np.array(L)
    print(f"  n = {n}: loss at T = {int(curves[n][-1, 0])}: {curves[n][-1, 1]:.3e}")
c1 = curves[1]
check("n = 1: linear (exponential) convergence: loss < 1e-20 within 3000 steps", np.any((c1[:, 0] <= 3000) & (c1[:, 1] < 1e-20)))
for n in (2, 3, 5):
    c = curves[n]; sel = c[:, 0] >= T / 4
    slope = np.polyfit(np.log(c[sel, 0]), np.log(c[sel, 1]), 1)[0]
    ratio = np.log(c[-1, 1]) / np.log(eta * c[-1, 0])
    print(f"  n = {n}: fitted log-log slope over T in [{T // 4}, {T}] = {slope:.2f}; log L / log(eta T) at the end = {ratio:.2f}")
    check(f"n = {n}: algebraic decay with slope near -3 (between -3.6 and -2.4)", -3.6 < slope < -2.4)
# 3-D slice: d = 2, v = e1, w1 = (a, b), w2 = (1 - a, -b)
v2 = np.array([1., 0.])
fs = lambda a, b: loss(np.array([[a, b], [1 - a, -b]]), v2)
bs = np.array([1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2])
vals = np.array([fs(.5, b) for b in bs])
p = np.polyfit(np.log(bs), np.log(vals), 1)[0]
check("on the slice, loss ~ |b|^3 near the valley (a = 0.5)", abs(p - 3) < .05, f"exponent {p:.3f}")
check("the valley b = 0, 0 <= a <= 1 is all global minima (loss 0)", max(fs(a, 0) for a in np.linspace(0, 1, 11)) < 1e-14)
gs = lambda a, b: fd_grad(lambda q: fs(q[0], q[1]), np.array([a, b]), 1e-7)
q = np.array([.3, .4]); Ls = []
for t in range(20001):
    if t % 20 == 0:
        Ls.append((t, fs(*q)))
    q = q - .5 * gs(*q)
Ls = np.array(Ls); sel = Ls[:, 0] >= 5000
ps = np.polyfit(np.log(Ls[sel, 0]), np.log(Ls[sel, 1]), 1)[0]
check("GD on the slice decays algebraically like t^-3", abs(ps + 3) < .3, f"slope {ps:.2f}")
out = {"eta": eta, "sigma": sigma, "d": d}
for n, c in curves.items():
    idx = np.unique(np.round(np.logspace(0, np.log10(len(c) - 1), 160)).astype(int))
    out[f"c{n}"] = [[float(c[i, 0]), float(f"{c[i, 1]:.4g}")] for i in idx]
save_json("xu2023over", "data.json", out)
done()
