"""Li, Wang, Li 2022 (survey key wang2022analyzing): analyzing sharpness along the GD trajectory.

Gram matrix M = (2/n) (dF/dtheta)(dF/dtheta)^T, residual D = F - Y, loss L = |D|^2 / n.
Eq. (2): for f = A^T h(W, x), gradient flow gives d|A|^2/dt = -(4/n) D^T F, so the output-layer norm grows iff D.F < 0.
Sec. 4 model: f(x) = (1/sqrt m) A^T W x (two-layer linear), labels +-1. Replicated offline: sharpness and |A|^2 along GD.
3D tab: scalar case m = d = n = 1, x = 1, y = 1: L(a, w) = (aw - 1)^2 (paper's normalisation L = |D|^2/n).
"""
from common import *

print("wang2022analyzing (Li, Wang, Li)")
L1 = lambda z: (z[0] * z[1] - 1) ** 2
G1 = lambda z: np.array([2 * (z[0] * z[1] - 1) * z[1], 2 * (z[0] * z[1] - 1) * z[0]])
check_grad("scalar (aw - 1)^2", L1, G1, rand_pts(5, -2, 2))
check_hess_at("on the minimum (2, 0.5): top eigenvalue 2(a^2 + w^2)", L1, [2, .5], [[2 * .25, 2 * 1], [2 * 1, 2 * 4]])
# Eq. (2) for the scalar model: d a^2/dt = -4 D F  (n = 1)
z = np.array([1.3, .4]); h = 1e-6
D = z[0] * z[1] - 1; F = z[0] * z[1]
z2 = z - h * G1(z)
check("Eq. (2): d|A|^2/dt = -(4/n) D.F (gradient flow, scalar case)", abs((z2[0] ** 2 - z[0] ** 2) / h - (-4 * D * F)) < 1e-4)

# ---------- two-layer linear network replication ----------
rng = np.random.default_rng(3)
n, d, m = 32, 48, 64
X = rng.normal(size=(n, d)) * np.linspace(1.6, .4, d)        # anisotropic inputs: a top eigen-direction of X^T X
Y = np.sign(X @ rng.normal(size=d) + .3 * rng.normal(size=n))
A0 = rng.normal(size=m) * .15; W0 = rng.normal(size=(m, d)) * .15 / np.sqrt(d)


def unpack(v):
    return v[:m], v[m:].reshape(m, d)


def loss_grad(v):
    A, W = unpack(v)
    H = X @ W.T                       # n x m
    Fv = H @ A / np.sqrt(m); Dv = Fv - Y
    Lv = Dv @ Dv / n
    gA = (2 / n) * H.T @ Dv / np.sqrt(m)
    gW = (2 / n) * np.outer(A, Dv @ X) / np.sqrt(m)
    return Lv, np.concatenate([gA, gW.ravel()]), Dv, Fv


v0 = np.concatenate([A0, W0.ravel()])
g = loss_grad(v0)[1]; err = 0
for i in rng.choice(v0.size, 10, replace=False):
    e = np.zeros_like(v0); e[i] = 1e-6
    err = max(err, abs((loss_grad(v0 + e)[0] - loss_grad(v0 - e)[0]) / 2e-6 - g[i]))
check("two-layer linear net gradient vs finite differences", err < 1e-8, f"max err {err:.1e}")


def hvp(v, u, h=1e-5):
    return (loss_grad(v + h * u)[1] - loss_grad(v - h * u)[1]) / (2 * h)


def sharp(v, u, iters):
    lam = 0
    for _ in range(iters):
        hu = hvp(v, u); lam = u @ hu; u = hu / np.linalg.norm(hu)
    return lam, u


def run(eta, T):
    v = v0.copy(); u = rng.normal(size=v.size); u /= np.linalg.norm(u); lam, u = sharp(v, u, 80); rows = []
    for t in range(T):
        Lv, gv, Dv, Fv = loss_grad(v)
        lam, u = sharp(v, u, 6)
        A, W = unpack(v)
        rows.append([t, Lv, lam, A @ A, Dv @ Fv / n])
        v = v - eta * gv
    return np.array(rows)


R0 = run(.01, 3000)
print(f"    eta=0.01 (gradient-flow-like): S0 {R0[0, 2]:.2f}, S max {R0[:, 2].max():.2f}, |A|^2 {R0[0, 3]:.2f} -> {R0[-1, 3]:.2f}, loss {R0[-1, 1]:.2e}")
ETA = None
# step sizes whose threshold 2/eta lies halfway (and 70%) between the initial and the gradient-flow sharpness
ETAS = [round(2 / (R0[0, 2] + f * (R0[:, 2].max() - R0[0, 2])), 3) for f in (.5, .7)]
print("    trying", ETAS)
for eta in ETAS:
    R = run(eta, 1200)
    ok = np.isfinite(R[:, 1]).all() and R[:, 2].max() > 2 / eta and R[-1, 1] < R[0, 1]
    print(f"    eta={eta}: S0 {R[0, 2]:.2f}, S max {R[:, 2].max():.2f} (2/eta {2 / eta:.2f}), late S {R[-300:, 2].mean():.2f}, loss {R[-1, 1]:.2e}, ok {ok}")
    if ok and ETA is None and R[0, 2] < 2 / eta * .8:
        ETA, RR = eta, R
check("progressive sharpening then EoS found for some eta", ETA is not None, f"eta = {ETA}")
R = RR
check("sharpness rises before first reaching 2/eta (progressive sharpening)", R[:int(np.argmax(R[:, 2] > 2 / ETA)), 2][-1] > 1.15 * R[0, 2])
late = R[int(np.argmax(R[:, 2] > 2 / ETA)) + 50:]
check("after crossing, sharpness oscillates around 2/eta", abs(late[:, 2].mean() - 2 / ETA) / (2 / ETA) < .15, f"mean {late[:, 2].mean():.3f} vs {2 / ETA:.3f}")
dS, dA = np.diff(R[:, 2]), np.diff(R[:, 3])
# low-frequency agreement (paper: 'most of the time', high-frequency oscillations not captured): smooth over 10 steps
k = 10
sS = np.convolve(dS, np.ones(k), 'valid'); sA = np.convolve(dA, np.ones(k), 'valid')
agree = np.mean(np.sign(sS) == np.sign(sA))
check("|A|^2 and sharpness move in the same direction most of the time (10-step windows)", agree > .7, f"agreement {agree:.2f} (per step {np.mean(np.sign(dS) == np.sign(dA)):.2f})")
# Eq. (2) sign rule in discrete GD: |A|^2 rises when D.F < 0
sgn = np.mean(np.sign(dA) == -np.sign(R[:-1, 4]))
check("discrete GD: |A|^2 increases exactly when D.F < 0 (to first order)", sgn > .9, f"agreement {sgn:.2f}")
save_json("wang2022analyzing", "data.json", {"eta": ETA, "rows": [[int(r[0]), float(f"{r[1]:.4g}"), round(float(r[2]), 4), round(float(r[3]), 4), float(f"{r[4]:.4g}")] for r in R],
                                             "setup": f"f(x) = A^T W x / sqrt(m), n = {n}, d = {d}, m = {m}, labels +-1, MSE |D|^2/n, full-batch GD"})
rows = [[int(r[0]), float(f"{r[1]:.4g}"), round(float(r[2]), 4), round(float(r[3]), 4)] for r in R]
Q = np.array(rows[:300]); a_ = np.mean([np.sign(Q[i + 10, 2] - Q[i, 2]) == np.sign(Q[i + 10, 3] - Q[i, 3]) for i in range(290)])
check("page readout: agreement over the first 300 stored steps", a_ > .9, f"{a_:.3f}")
done()
