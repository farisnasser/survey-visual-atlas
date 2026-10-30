"""Lu, Ma, Lu, Lu, Ying 2020: mean-field (continuous-depth) ResNet. Discretized model (Sec. 4 / Algorithm 1):
X_0 = w2 x = (x, 0),  X_{n+1} = X_n + (1/L)(1/M) sum_j a_{nj} tanh(w_{nj} . X_n + b_{nj}),  output <w1, X_L>, w1 = (1, 0) fixed,
E = mean_x 1/2 (<w1, X_L(x)> - y(x))^2,  y(x) = sin(pi x) on [-1, 1].  Particles theta_{nj} = (a in R^2, w in R^2, b), GD on all
of them with step eta * L * M (time-rescaled particle gradient flow of the mean-field model).
Adjoint p_n = dE_x / dX_n (backprop through depth): Lemma 1 (ICML) says |p(., t)|_mu >= e^{-(C1 + C2 r)} E(rho) for all t,
i.e. the gradient signal does not vanish through depth while the loss is positive.
"""
from common import *

print("lu2020mean")
n = 48
xs = np.linspace(-1, 1, n); ys = np.sin(np.pi * xs)
w1 = np.array([1.0, 0.0])


def init(L, M, seed):
    R = np.random.default_rng(seed)
    return [R.normal(0, 1, (L, M, 2)), R.normal(0, 1, (L, M, 2)), R.normal(0, 1, (L, M))]   # a, w, b


def forward(P, L, M):
    a, w, b = P
    X = np.zeros((L + 1, n, 2)); X[0, :, 0] = xs
    for k in range(L):
        z = X[k] @ w[k].T + b[k][None, :]                       # (n, M)
        X[k + 1] = X[k] + np.tanh(z) @ a[k] / (L * M)
    return X


def loss_grad(P, L, M):
    a, w, b = P
    X = forward(P, L, M)
    r = X[L] @ w1 - ys
    E = .5 * np.mean(r ** 2)
    p = np.zeros((L + 1, n, 2)); p[L] = r[:, None] * w1[None, :] / n
    ga, gw, gb = np.zeros_like(a), np.zeros_like(w), np.zeros_like(b)
    for k in range(L - 1, -1, -1):
        z = X[k] @ w[k].T + b[k][None, :]; t = np.tanh(z); s = 1 - t * t
        ga[k] = t.T @ p[k + 1] / (L * M)
        q = (p[k + 1] @ a[k].T) * s / (L * M)                    # (n, M) = dE/dz
        gw[k] = q.T @ X[k]; gb[k] = q.sum(0)
        p[k] = p[k + 1] + q @ w[k]
    return E, [ga, gw, gb], X, p


L0, M0 = 3, 4
P0 = init(L0, M0, 1)
flat = lambda P: np.concatenate([q.ravel() for q in P])
def unflat(v, L, M):
    s1, s2 = L * M * 2, L * M * 4
    return [v[:s1].reshape(L, M, 2), v[s1:s2].reshape(L, M, 2), v[s2:].reshape(L, M)]
check_grad("E(theta) by backprop through depth (L = 3, M = 4)", lambda v: loss_grad(unflat(v, L0, M0), L0, M0)[0],
           lambda v: flat(loss_grad(unflat(v, L0, M0), L0, M0)[1]), [flat(P0), .5 * flat(P0)])


def train(L, M, seed=0, steps=3000, eta=.6, keep=0):
    P = init(L, M, seed); hist, frames, adj = [], [], []
    for s in range(steps + 1):
        E, G, X, p = loss_grad(P, L, M)
        hist.append(E)
        if keep and s % keep == 0:
            frames.append(r(X[:, ::2, :], 3))
            pn = np.sqrt(np.sum(p ** 2, axis=(1, 2)) * n)           # |p(., t)|_mu (mean-square over data, times sqrt n)
            adj.append(sig(pn / max(pn[-1], 1e-300), 4))
        if s < steps:
            P = [q - eta * L * M * g for q, g in zip(P, G)]
    return np.array(hist), frames, adj, P


H16, F16, A16, P16 = train(16, 32, keep=300)
print(f"  L = 16, M = 32: E {H16[0]:.4f} -> {H16[-1]:.2e}")
check("training drives the loss near zero (mean-field ResNet, L = 16, M = 32)", H16[-1] < 1e-3 * H16[0])
ratios = [min(a_) for a_ in A16[:-1]]
print("  min_t |p(t)| / |p(1)| at the saved training times:", [round(x, 3) for x in ratios])
check("Lemma 1: the adjoint never collapses through depth (|p(t)| >= 0.3 |p(1)| at every depth and checkpoint)", min(ratios) > .3)
curves = {}
for L in (4, 16, 64):
    H, *_ = train(L, 32, steps=1500)
    curves[L] = H
    print(f"  L = {L:2d}: E after 1500 steps {H[-1]:.2e}")
check("continuum limit: loss curves for L = 16 and L = 64 nearly coincide (same 1/L scaling)", np.max(np.abs(np.log(curves[64][::50] + 1e-12) - np.log(curves[16][::50] + 1e-12))) < 1.0)
check("notes' Euler example: (1 + 1/2)^2 = 2.25, (1 + 1/10)^10 = 2.594 -> e", abs(1.5 ** 2 - 2.25) < 1e-12 and abs(1.1 ** 10 - 2.5937) < 1e-4)
save_json("lu2020mean", "data.json", {"xs": r(xs[::2], 4), "ys": r(ys[::2], 4), "frames": F16, "adj": A16, "H16": sig(H16[::10], 4),
                                     "curves": {str(L): sig(curves[L][::10], 4) for L in curves}})
done()
