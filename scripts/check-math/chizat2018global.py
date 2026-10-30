"""Chizat & Bach 2018: many-particle gradient flow of a one-hidden-layer ReLU net reaches the global minimum.

Replicates the setup of Fig. 2: inputs x ~ N(0, I_2), teacher y(x) = (1/4) sum_k w*_k relu(theta*_k . x) with m0 = 4 neurons,
student f(x) = (1/m) sum_j w_j relu(theta_j . x) (mean-field scaling), particles (w_j, theta_j) initialized on a small sphere,
gradient descent with step m*eta on each particle (time-rescaled particle gradient flow), square loss on a fixed sample.
Paper: failure with 5 particles, success with 10 and 100 (its random teacher). On our fixed teacher m = 5 and m = 10 stall at a
non-global point (loss ~3e-5, unchanged after 4x more steps), while m = 20 and m = 100 go below 1e-6 and keep decreasing:
the same phenomenon (a modest over-parameterization suffices) with an instance-dependent threshold. We plot |w_j| theta_j.
"""
from common import *

print("chizat2018global")
N = 600
RX = np.random.default_rng(29)
X = RX.normal(size=(N, 2))
ang = np.array([.3, 1.9, 3.4, 4.9])
TH = np.stack([np.cos(ang), np.sin(ang)], 1) * np.array([[1.4], [1.0], [1.2], [.9]])
WS = np.array([1.5, -1.2, 1.3, 1.0])
relu = lambda z: np.maximum(z, 0)
Y = relu(X @ TH.T) @ WS / 4


def loss(w, th):
    return .5 * np.mean((relu(X @ th.T) @ w / len(w) - Y) ** 2)


def grads(w, th):
    m = len(w); Z = X @ th.T; A = relu(Z); r = A @ w / m - Y
    gw = A.T @ r / N / m
    gth = ((r[:, None] * (Z > 0) * w[None, :]).T @ X) / N / m
    return gw, gth


w0 = np.array([.3, -.2, .1]); t0 = RX.normal(size=(3, 2))
p0 = np.concatenate([w0, t0.ravel()])
fl = lambda p: loss(p[:3], p[3:].reshape(3, 2))
gl = lambda p: np.concatenate([grads(p[:3], p[3:].reshape(3, 2))[0], grads(p[:3], p[3:].reshape(3, 2))[1].ravel()])
check_grad("loss (w, theta), 3 particles", fl, gl, [p0, p0 * .7 + .1, -p0])
check("teacher is realizable: loss of the teacher = 0", loss(WS * 1.0 * 4 / 4, TH) < 1e-30 or abs(loss(np.repeat(WS, 1), TH)) < 1e-30)


def train(m, seed, steps=5000, eta=.4, r0=.05, save_every=0):
    R = np.random.default_rng(seed)
    a = 2 * np.pi * (np.arange(m) + R.uniform()) / m
    th = r0 * np.stack([np.cos(a), np.sin(a)], 1)
    w = r0 * np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
    traj, L = [], []
    for k in range(steps):
        gw, gth = grads(w, th)
        w, th = w - eta * m * gw, th - eta * m * gth
        if save_every and k % save_every == 0:
            # u_j = |w_j| theta_j and sign(w_j): f = (1/m) sum sign_j relu(u_j . x)
            traj.append(np.c_[np.abs(w)[:, None] * th, np.sign(w)]); L.append(loss(w, th))
    return w, th, loss(w, th), traj, L


res = {m: [train(m, s)[2] for s in range(2)] for m in (5, 10, 20, 100)}
for m, v in res.items():
    print(f"  m = {m:3d}: final loss over 2 inits  " + ", ".join(f"{x:.2e}" for x in v))
L0 = loss(np.zeros(1), np.zeros((1, 2)))
print(f"  loss of f = 0: {L0:.4f}")
check("m = 5 and m = 10 stall at a non-global point (loss > 1e-5)", min(res[5] + res[10]) > 1e-5)
check("the stall is a plateau, not slow progress: m = 10 loss after 20000 steps within 5% of 5000 steps", abs(train(10, 0, steps=20000)[2] / res[10][0] - 1) < .05)
check("m = 20 and m = 100 reach the global minimum within optimization error (loss < 2e-6)", max(res[20] + res[100]) < 2e-6)
out = {"TH": r(TH, 4), "WS": r(WS, 4), "L0": round(float(L0), 6)}
for m in (5, 10, 20, 100):
    w, th, Lf, traj, L = train(m, 0, save_every=25)
    out[f"traj{m}"] = [r(t, 3) for t in traj[::2]]; out[f"loss{m}"] = sig(np.maximum(L, 1e-12), 3); out[f"final{m}"] = float(f"{Lf:.3g}")
check("loss curves saved", all(len(out[f"loss{m}"]) == 200 for m in (5, 10, 20, 100)))
save_json("chizat2018global", "data.json", out)
done()
