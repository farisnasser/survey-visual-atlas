"""Li, Ma, Zhang 2020: a trained two-layer net learns f*(x) = sum_i a_i |w*_i . x| by moving its features (beyond NTK).

Setup (Sec. 1.1) with d = 3, W* = I, a_i = 1/3 (kappa = 1): x ~ N(0, I), student f_W(x) = (1/m) sum_i |w_i| relu(w_i . x) (Eq. 1.2).
Population loss is exact through the arc-cosine kernel  E relu(u.x) relu(v.x) = (|u||v| sin(phi) + (pi - phi) u.v) / (2 pi),
and |t| = relu(t) + relu(-t).  Algorithm 1: truncated GD (no update for neurons with |w|^2 > 1/lambda), init w ~ N(0, I/d).
Baseline (illustration of the fixed-feature limitation, not Theorem 1.2 itself): best linear combination of the m random
initial features relu(w_i(0) . x) (a random-feature kernel method), solved exactly in the population.
"""
from common import *

print("li2020learning")
d, m = 3, 60
a = np.full(d, 1 / d); E = np.eye(d)


def k(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    c = np.clip(u @ v / (nu * nv), -1, 1); ph = np.arccos(c)
    return (nu * nv * np.sin(ph) + (np.pi - ph) * (u @ v)) / (2 * np.pi)


def gk(u, v):   # gradient of k in u
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    c = np.clip(u @ v / (nu * nv), -1, 1); ph = np.arccos(c)
    return ((np.pi - ph) * v + nv * np.sin(ph) * u / nu) / (2 * np.pi)


Rm = np.random.default_rng(0)
xs = Rm.normal(size=(400000, d)); relu = lambda z: np.maximum(z, 0)
u0, v0 = Rm.normal(size=d), Rm.normal(size=d)
check("arc-cosine kernel matches Monte Carlo", abs(k(u0, v0) - np.mean(relu(xs @ u0) * relu(xs @ v0))) < 1e-2 * (1 + abs(k(u0, v0))))
check_grad("kernel gradient", lambda u: k(u, v0), lambda u: gk(u, v0), [u0, -u0 + .3])
T = [(s * E[i], a[i]) for i in range(d) for s in (1, -1)]          # teacher as 2d ReLU neurons with weights a_i
Ftt = sum(ai * aj * k(ui, uj) for ui, ai in T for uj, aj in T)       # E f*^2


TU = np.array([u for u, _ in T]); TA = np.array([aj for _, aj in T])


def kmat(A, B):
    nA, nB = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    C = np.clip((A / nA[:, None]) @ (B / nB[:, None]).T, -1, 1); P = np.arccos(C)
    return nA, nB, C, P, np.outer(nA, nB) * (np.sin(P) + (np.pi - P) * C) / (2 * np.pi)


def loss(W):
    m_ = len(W); n, _, _, _, K = kmat(W, W); _, _, _, _, KT = kmat(W, TU)
    return n @ K @ n / m_ ** 2 - 2 * n @ (KT @ TA) / m_ + Ftt


def grad(W):
    # d/dw_i of (1/m^2) sum_jl n_j n_l k(w_j, w_l) = (2/m^2) sum_j n_j (k(w_i, w_j) h_i + n_i grad_u k(w_i, w_j)) (the j = i term
    # has the same form), with grad_u k(u, v) = ((pi - phi) v + |v| sin(phi) u/|u|)/(2 pi).
    m_ = len(W); n, _, _, P, K = kmat(W, W); h = W / n[:, None]
    gss = (K @ n)[:, None] * h + n[:, None] * (((np.pi - P) * n[None, :]) @ W + ((np.sin(P) * n[None, :]) @ n)[:, None] * h) / (2 * np.pi)
    _, nT, _, PT, KT = kmat(W, TU)
    gst = (KT @ TA)[:, None] * h + n[:, None] * (((np.pi - PT) * TA[None, :]) @ TU + ((np.sin(PT) * TA[None, :]) @ nT)[:, None] * h) / (2 * np.pi)
    return 2 * gss / m_ ** 2 - 2 * gst / m_


W0 = Rm.normal(0, 1 / np.sqrt(d), (4, d))
check_grad("population loss gradient (m = 4)", lambda w: loss(w.reshape(4, d)), lambda w: grad(w.reshape(4, d)).ravel(), [W0.ravel(), .5 * W0.ravel() + .1])
Xmc = Rm.normal(size=(300000, d)); fstar = np.abs(Xmc) @ a
fW = lambda W: (relu(Xmc @ W.T) * np.linalg.norm(W, axis=1)).mean(1)
check("exact population loss matches Monte Carlo", abs(loss(W0) - np.mean((fW(W0) - fstar) ** 2)) < 2e-3, f"{loss(W0):.4f}")

R = np.random.default_rng(91)
W = R.normal(0, 1 / np.sqrt(d), (m, d)); Winit = W.copy()
eta, lam, steps = 2.0, 1 / 25, 6000
hist, frames = [loss(W)], [r(W, 4)]
for t in range(steps):
    G = grad(W); mask = (np.sum(W * W, 1) <= 1 / lam)[:, None]
    W = W - eta * mask * G
    hist.append(loss(W))
    if (t + 1) % 150 == 0:
        frames.append(r(W, 4))
hist = np.array(hist)
print(f"  trained network: loss {hist[0]:.4f} -> {hist[-1]:.2e}")
# alignment: fraction of neuron mass within 15 degrees of some +-e_i
nn = np.linalg.norm(W, axis=1); cosmax = np.max(np.abs(W / nn[:, None]), 1)
mass = np.sum(nn ** 2 * (cosmax > np.cos(np.radians(15)))) / np.sum(nn ** 2)
check("trained network fits the teacher (loss < 2e-4, from ~0.23)", hist[-1] < 2e-4 and hist[0] > .05)
check("the features moved toward the teacher axes: > 60% of the neuron mass within 15 deg of some +-e_i (at init: a few %)", mass > .6, f"{mass:.1%}")
n0 = np.linalg.norm(Winit, axis=1); m0 = np.sum(n0 ** 2 * (np.max(np.abs(Winit / n0[:, None]), 1) > np.cos(np.radians(15)))) / np.sum(n0 ** 2)
print(f"  mass within 15 deg of an axis at init: {m0:.1%}")
# fixed random features at the same width (kernel method on the initial features), exact in the population
K = np.array([[k(u, v) for v in Winit] for u in Winit]); b = np.array([sum(aj * k(u, uj) for uj, aj in T) for u in Winit])
c = np.linalg.solve(K + 1e-10 * np.eye(m), b); rf = Ftt - b @ c
print(f"  best fixed random-feature model with the same {m} features: loss {rf:.2e}")
check("same-width fixed random features are clearly worse than the trained network (> 5x)", rf > 5 * hist[-1])
# plateau then drop (stage-wise dynamics): time to reach 10% vs 1% of the initial loss
t10 = int(np.argmax(hist < .1 * hist[0])); t1 = int(np.argmax(hist < .01 * hist[0]))
print(f"  steps to 10% of the initial loss: {t10}, to 1%: {t1}")
save_json("li2020learning", "data.json", {"hist": sig(hist[::20], 4), "frames": frames, "rf": float(f"{rf:.4g}"), "mass": round(float(mass), 4), "a": a.tolist(),
                                          "eta": eta, "lam": lam, "t10": t10, "t1": t1, "m0": round(float(m0), 4)})
done()
