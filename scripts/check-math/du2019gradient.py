"""Du et al. 2019: two-layer softplus net, NTK scaling, full-batch GD; Gram-matrix bound; width stability.

The browser runs the same model live (papers/du2019gradient/visual.js); this script checks the claims it shows.
Inputs x = (cos t, sin t, 1)/sqrt(2) have ||x|| = 1 as the theorem assumes (the constant coordinate acts like a bias).
"""
from common import *

print("du2019gradient")
softplus = lambda z: np.logaddexp(0, z)
sig = lambda z: 1 / (1 + np.exp(-z))
TH = np.array([.3, 2.4, 4.4])
Y = np.array([.8, -.5, .6])


def data(n=3):
    X = np.column_stack([np.cos(TH[:n]), np.sin(TH[:n]), np.ones(n)]) / np.sqrt(2)
    return X, Y[:n]


def init(m, seed):
    r = np.random.default_rng(seed)
    return r.normal(size=(m, 3)), r.normal(size=m)


def predict(W, a, X):
    return softplus(X @ W.T) @ a / np.sqrt(len(a))


def jac(W, a, X):
    m = len(a)
    Z = X @ W.T
    JW = (sig(Z) * a[None, :] / np.sqrt(m))[:, :, None] * X[:, None, :]
    return np.concatenate([JW.reshape(len(X), -1), softplus(Z) / np.sqrt(m)], 1)


def train(m, X, y, steps, seed=0, eta=None):
    W, a = init(m, seed)
    W0 = W.copy()
    J = jac(W, a, X)
    ev0 = eig(J @ J.T)
    eta = 1 / ev0[-1] if eta is None else eta
    loss, lmin, res = [], [], []
    for k in range(steps):
        rr = predict(W, a, X) - y
        loss.append(.5 * rr @ rr)
        res.append(rr.copy())
        if k % 10 == 0:
            J = jac(W, a, X)
            lmin.append(eig(J @ J.T)[0])
        Z = X @ W.T
        ga = softplus(Z).T @ rr / np.sqrt(m)
        gW = ((sig(Z) * a[None, :] / np.sqrt(m)) * rr[:, None]).T @ X
        W, a = W - eta * gW, a - eta * ga
    return dict(loss=np.array(loss), lmin=np.array(lmin), res=np.array(res), lam0=ev0[0], eta=eta,
                move=np.linalg.norm(W - W0) / np.linalg.norm(W0))


X, y = data()
W, a = init(32, 0)
fl = lambda t: .5 * np.sum((predict(t[:96].reshape(32, 3), t[96:], X) - y) ** 2)
th = np.concatenate([W.ravel(), a])
check_grad("loss via Jacobian", fl, lambda t: jac(t[:96].reshape(32, 3), t[96:], X).T @ (predict(t[:96].reshape(32, 3), t[96:], X) - y), [th])
out = {}
for m in [8, 64, 1024, 8192]:
    o = train(m, X, y, 300)
    bound = (1 - o["eta"] * o["lam0"] / 2) ** np.arange(300) * o["loss"][0]
    drift = np.max(np.abs(o["lmin"] / o["lam0"] - 1))
    out[m] = (o, bound, drift)
    print(f"  m={m:5d}: lambda0={o['lam0']:.3f} eta={o['eta']:.3f} L(300)={o['loss'][-1]:.1e} below bound={np.all(o['loss'] <= bound * (1 + 1e-9))} "
          f"movement {o['move']:.4f} max |lambda_min(G_k)/lambda0 - 1| {drift:.3f}")
check("wide net: loss below (1 - eta*lambda0/2)^k L(0) at every k", np.all(out[8192][0]["loss"] <= out[8192][1] * (1 + 1e-9)))
check("wide net: Gram matrix stays close to G(0)", out[8192][2] < .1)
check("narrow net: Gram matrix drifts much more", out[8][2] > 3 * out[8192][2])
check("weights move less when wider", out[8192][0]["move"] < out[8][0]["move"] / 10)
X2, y2 = data(2)
for m in [4, 4096]:
    o = train(m, X2, y2, 200, seed=2)
    R = o["res"]
    ch = R[-1] - R[0]
    D = R - R[0]
    dev = np.max(np.abs(ch[0] * D[:, 1] - ch[1] * D[:, 0])) / np.linalg.norm(ch) ** 2
    print(f"  n=2 prediction space, m={m}: final loss {o['loss'][-1]:.1e}, path deviation from the chord {dev:.3f}")
dev = {}
for m in [4, 4096]:
    W, a = init(m, 2)
    J = jac(W, a, X2)
    o = train(m, X2, y2, 200, seed=2)
    M = np.eye(2) - o["eta"] * (J @ J.T)
    lin = np.array([np.linalg.matrix_power(M, k) @ o["res"][0] for k in range(200)])
    dev[m] = np.max(np.linalg.norm(o["res"] - lin, axis=1)) / np.linalg.norm(o["res"][0])
    print(f"  m={m}: max |actual residual - (I - eta G0)^k r0| / |r0| = {dev[m]:.4f}")
check("wide net follows the linear Gram dynamics; narrow net does not", dev[4096] < .002 and dev[4] > .05)
done()
