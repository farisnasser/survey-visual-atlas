"""Mei, Misiakiewicz, Montanari 2019: the mean-field approximation error of SGD is ~1/sqrt(N), independent of dimension.

Sec. 3.2 example: y = +-1 w.p. 1/2, x ~ N(0, Sigma_y), Sigma_+- = diag((1 +- Delta)^2 I_{s0}, I_{d-s0}) (U = I), s0 = d/2.
Network with fixed coefficients: f_N(x) = (1/N) sum_i sigma(<w_i, x>), sigma = truncated ReLU (t1, t2, s1, s2 our choice).
One-pass SGD (Eq. SGD): w_i <- w_i + 2 eps (y_k - f_N(x_k)) sigma'(<w_i, x_k>) x_k, time t = k eps, eps = 0.05/d (Theorem 1 needs eps << 1/D).
Init: |w| ~ Unif[0.8, 1.2], direction uniform on S^{d-1} (an uninformative initialization).
Theorem 1(A): sup_k |R_N(theta_k) - R(rho_{k eps})| <= K e^{KT} (sqrt(log N) + z)/sqrt(N) + K e^{KT} (sqrt(D + log N) + z) sqrt(eps).
The limit R(rho_t) itself depends on d (s0 = d/2 informative directions); the claim is about the N-dependence of the error.
We use N = 3200 as a proxy for the PDE solution and measure the deviation for smaller N at d = 20 and d = 80.
"""
from common import *

print("mei2019mean")
DEL, T1, T2, S1, S2 = .6, .5, 1.5, -2.5, 7.5
sig_ = lambda t: np.where(t <= T1, S1, np.where(t >= T2, S2, S1 + (S2 - S1) * (t - T1) / (T2 - T1)))
dsig = lambda t: np.where((t > T1) & (t < T2), (S2 - S1) / (T2 - T1), 0.0)
check_grad("truncated ReLU slope (smooth part)", lambda z: float(sig_(z[0] + 1.0)), lambda z: np.array([float(dsig(z[0] + 1.0))]), [np.array([-.3]), np.array([.2])])


def sample(R, n, d):
    s0 = d // 2; y = R.choice([-1., 1.], n); X = R.normal(size=(n, d))
    X[:, :s0] *= (1 + DEL * y)[:, None]
    return X, y


def risk(W, Xt, yt):
    return float(np.mean((yt - sig_(Xt @ W.T).mean(1)) ** 2))


def run(N, d, seed, Tt=3.0, every=None):
    R = np.random.default_rng(seed); eps = .05 / d; K = int(Tt / eps)
    W = R.normal(size=(N, d)); W *= (R.uniform(.8, 1.2, N) / np.linalg.norm(W, axis=1))[:, None]
    Xt, yt = sample(np.random.default_rng(999), 1500, d)
    every = every or max(1, K // 30)
    ts, Rs, snaps = [], [], []
    for k in range(K + 1):
        if k % every == 0:
            ts.append(k * eps); Rs.append(risk(W, Xt, yt))
            s0 = d // 2; snaps.append(np.c_[np.linalg.norm(W[:, :s0], axis=1), np.linalg.norm(W[:, s0:], axis=1)])
        if k == K:
            break
        x, yy = sample(R, 1, d); x = x[0]; z = W @ x
        W = W + 2 * eps * (yy[0] - sig_(z).mean()) * dsig(z)[:, None] * x[None, :]
    return np.array(ts), np.array(Rs), snaps, W


out = {"Ns": [50, 200, 800], "ds": [20, 80]}
dev = {}
for d in (20, 80):
    tR, Rref, snR, _ = run(3200, d, 0)
    out[f"t{d}"] = r(tR, 3); out[f"ref{d}"] = sig(Rref, 4)
    for N in (50, 200, 800):
        devs = []
        for s in range(3):
            t_, R_, sn, _ = run(N, d, 100 + s)
            devs.append(np.max(np.abs(R_ - Rref)))
            if s == 0:
                out[f"R{d}_{N}"] = sig(R_, 4)
        dev[(d, N)] = float(np.mean(devs))
        print(f"  d = {d:3d}, N = {N:4d}: mean sup_t |R_N - R_3200| = {dev[(d, N)]:.4f}")
    print(f"  d = {d}: risk {Rref[0]:.3f} -> {Rref[-1]:.3f} (N = 3200)")
    out[f"snap{d}"] = [r(s_[::8], 3) for s_ in snR[::4]]
check("risk decreases well below its initial value (learning happens)", out["ref20"][-1] < .6 * out["ref20"][0] and out["ref80"][-1] < .6 * out["ref80"][0])
check("deviation from the large-N proxy shrinks with N (roughly 1/sqrt N)", all(dev[(d, 800)] < .7 * dev[(d, 50)] for d in (20, 80)))
check("dimension-free: at each N the deviation at d = 80 is within 2x of d = 20", all(dev[(80, N)] < 2 * dev[(20, N)] for N in (50, 200, 800)))
# R_inf(r1, r2): d -> infinity risk when every neuron has |P_V w| = r1, |P_perp w| = r2 and a uniform direction.
# Given y, <w, x> -> N(0, (1 + Delta y)^2 r1^2 + r2^2) and f concentrates on f_y = E sigma(Z sqrt(...)).
# exact: sigma(t) = S1 + (S2 - S1)/(T2 - T1) (relu(t - T1) - relu(t - T2)),  E relu(sZ - a) = s phi(a/s) - a (1 - Phi(a/s))
from math import erf, exp, pi, sqrt
Phi = np.vectorize(lambda z: .5 * (1 + erf(z / sqrt(2))))
phi = lambda z: np.exp(-z * z / 2) / np.sqrt(2 * pi)
Erelu = lambda s_, a_: s_ * phi(a_ / s_) - a_ * (1 - Phi(a_ / s_))
Esig = lambda s_: S1 + (S2 - S1) / (T2 - T1) * (Erelu(s_, T1) - Erelu(s_, T2))
Zmc = np.random.default_rng(0).normal(size=2_000_000)
check("closed form E sigma(sZ) matches Monte Carlo", all(abs(float(Esig(np.float64(s_))) - sig_(Zmc * s_).mean()) < 6e-3 for s_ in (.5, 1., 1.7)))
f_y = lambda yv, r1, r2: float(Esig(np.float64(np.sqrt((1 + DEL * yv) ** 2 * r1 ** 2 + r2 ** 2))))
Rinf = lambda r1, r2: .5 * sum((yv - f_y(yv, r1, r2)) ** 2 for yv in (-1, 1))
Rm = np.random.default_rng(5)
def risk_d(r1, r2, dd, n=4000):   # infinite N at finite d: f(x) = E_Z sigma(Z sqrt(r1^2 |x_V|^2/s0 + r2^2 |x_perp|^2/(d - s0)))
    Xt, yt = sample(np.random.default_rng(3), n, dd); s0 = dd // 2
    v = r1 ** 2 * np.sum(Xt[:, :s0] ** 2, 1) / s0 + r2 ** 2 * np.sum(Xt[:, s0:] ** 2, 1) / (dd - s0)
    f = Esig(np.sqrt(v))
    return float(.5 * sum(np.mean((yt[yt == c] - f[yt == c]) ** 2) for c in (-1, 1)))   # class-balanced: P(y = +-1) = 1/2 exactly
for r1, r2 in ((1.0, .5), (.4, 1.2)):
    dd, NN = 400, 1500
    W = Rm.normal(size=(NN, dd)); s0 = dd // 2
    W[:, :s0] *= r1 / np.linalg.norm(W[:, :s0], axis=1, keepdims=True); W[:, s0:] *= r2 / np.linalg.norm(W[:, s0:], axis=1, keepdims=True)
    Xt, yt = sample(np.random.default_rng(3), 4000, dd)
    a = float(.5 * sum(np.mean((yt[yt == c_] - sig_(Xt[yt == c_] @ W.T).mean(1)) ** 2) for c_ in (-1, 1)))
    b, c = risk_d(r1, r2, 400), risk_d(r1, r2, 20000, 1500)
    check(f"(r1, r2) = ({r1}, {r2}): N = 1500 network at d = 400 matches the infinite-N formula", abs(a - b) < .02, f"{a:.4f} vs {b:.4f}")
    check(f"(r1, r2) = ({r1}, {r2}): the finite-d formula tends to R_inf as d grows (d = 20000)", abs(c - Rinf(r1, r2)) < .01 and abs(c - Rinf(r1, r2)) < abs(b - Rinf(r1, r2)), f"d=400 {b:.4f}, d=20000 {c:.4f}, R_inf {Rinf(r1, r2):.4f}")
G = np.linspace(0, 3, 61)
grid = np.array([[Rinf(a, b) for a in G] for b in G])
j, i = np.unravel_index(np.argmin(grid), grid.shape)
print(f"  R_inf minimum on the grid: {grid[j, i]:.4f} at (r1, r2) = ({G[i]:.2f}, {G[j]:.2f})")
out["dev"] = {f"{d}_{N}": round(v, 5) for (d, N), v in dev.items()}
out["consts"] = [DEL, T1, T2, S1, S2]
save_json("mei2019mean", "data.json", out)
done()
