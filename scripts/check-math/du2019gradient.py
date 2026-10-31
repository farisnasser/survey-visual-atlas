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

# in-page notes (papers/du2019gradient/notes.md): fixed-Gram teaching model (Sec. 5), depth factors (Sec. 7), primer
Kt = np.diag([1., .5]); rk = np.array([1., -1]); rs, Ls_ = [rk], [.5 * rk @ rk]
for k in range(3):
    rk = (np.eye(2) - 1 * Kt) @ rk; rs.append(rk); Ls_.append(.5 * rk @ rk)
check("notes: r1 = (0,-1/2), r2 = (0,-1/4), r3 = (0,-1/8)", np.allclose(rs[1:], [[0, -.5], [0, -.25], [0, -.125]]))
check("notes: losses 1.00000, 0.12500, 0.03125, 0.00781", np.allclose(Ls_, [1, .125, .03125, .0078125]) and f"{Ls_[3]:.5f}" == "0.00781")
check("notes: smallest eigenvalue of K is 0.5", np.isclose(np.linalg.eigvalsh(Kt).min(), .5))
check("notes: 2^10 = 1024 and (1 + 1/10)^10 ~ 2.59 -> e", 2 ** 10 == 1024 and abs(1.1 ** 10 - 2.59) < .005 and abs((1 + 1e-6) ** 1e6 - np.e) < 1e-5)
Jl = np.array([[1., 0], [1, 1]]); Gl = Jl @ Jl.T
wl = np.linalg.eigvalsh(Gl)
check("primer: G = [[1,1],[1,2]], eigenvalues (3 -+ sqrt5)/2 ~ 0.382, 2.618", np.allclose(Gl, [[1, 1], [1, 2]]) and np.allclose(wl, [(3 - np.sqrt(5)) / 2, (3 + np.sqrt(5)) / 2]) and np.allclose(wl, [.382, 2.618], atol=5e-4))
eta_l = 1 / wl.max()
check("primer: eta = 1/lambda_max ~ 0.382; factors 0 and 1 - 0.382^2 ~ 0.854", abs(eta_l - .382) < 5e-4 and abs(1 - eta_l * wl.max()) < 1e-12 and abs(1 - eta_l * wl.min() - .854) < 5e-4)
# linear model: GD on theta moves predictions exactly by -eta G r
th0 = np.array([.3, -.7]); yv = np.array([1., 2]); r0_ = Jl @ th0 - yv
th1 = th0 - .1 * Jl.T @ r0_
check("primer: linear model, one GD step: r1 = (I - eta G) r0", np.allclose(Jl @ th1 - yv, (np.eye(2) - .1 * Gl) @ r0_))
coins = np.random.default_rng(5).choice([-1, 1], (20000, 100)).mean(1)
check("primer: average of 100 fair +-1 coins has std ~0.1 (1/sqrt(m))", abs(coins.std() - .1) < .003 and abs(1 / np.sqrt(1e4) - .01) < 1e-15)
done()
