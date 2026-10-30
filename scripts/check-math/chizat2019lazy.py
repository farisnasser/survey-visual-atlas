"""Chizat, Oyallon, Bach 2019: lazy training is a matter of scale.

(1) Theorem 2.2 on a 2-parameter model: h(w) = w1 + w2/2 + w1 w2, h(0) = 0, target y* = 1, R(y) = (y - y*)^2/2,
    F_a(w) = R(a h(w)) / a^2, linearized model h_bar(w) = Dh(0) w. Gradient descent with a small step approximates the
    gradient flows on [0, T]; we check sup |w_a - w0| = O(1/a) and sup |w_a - w_bar_a| = O(1/a^2) by fitting slopes.
(2) Fig. 1 (a)-(b): two-layer ReLU student f(x) = sum_j a_j relu(b_j . x), m = 20, symmetrized init (h(w0) = 0),
    weights N(0, tau^2); teacher with m0 = 3 neurons, |a_j b_j| = 1; n = 15 inputs uniform on the unit circle; square loss.
    Small tau (0.1): neurons move to the teacher directions; large tau (5): lazy, weights barely move.
"""
from common import *

print("chizat2019lazy")
h = lambda w: w[0] + .5 * w[1] + w[0] * w[1]
Dh = lambda w: np.array([1 + w[1], .5 + w[0]])
check_grad("Dh", h, Dh, rand_pts(10, -1, 1))
D0 = Dh(np.zeros(2))
Fa = lambda w, a: .5 * (a * h(w) - 1) ** 2 / a ** 2
gF = lambda w, a: (a * h(w) - 1) * Dh(w) / a
gFbar = lambda w, a: (a * (D0 @ w) - 1) * D0 / a
check_grad("F_alpha (alpha = 3)", lambda w: Fa(w, 3.), lambda w: gF(w, 3.), rand_pts(10, -1, 1))
w0 = np.array([.0, .0]); T, dt = 6.0, 1e-3


def flows(a):
    w, wb = w0.copy(), w0.copy(); dmax = emax = 0
    for _ in range(int(T / dt)):
        w = w - dt * gF(w, a); wb = wb - dt * gFbar(wb, a)
        dmax = max(dmax, np.linalg.norm(w - w0)); emax = max(emax, np.linalg.norm(w - wb))
    return dmax, emax, a * h(w)


alphas = np.array([2., 4., 8., 16., 32., 64.])
res = np.array([flows(a) for a in alphas])
s1 = np.polyfit(np.log(alphas), np.log(res[:, 0]), 1)[0]; s2 = np.polyfit(np.log(alphas[2:]), np.log(res[2:, 1]), 1)[0]
for a, (dm, em, out) in zip(alphas, res):
    print(f"  alpha = {a:4.0f}: sup|w - w0| = {dm:.3e}, sup|w - w_bar| = {em:.3e}, output a h(w(T)) = {out:.4f}")
check("Theorem 2.2: sup |w_a - w0| = O(1/alpha) (log-log slope ~ -1)", abs(s1 + 1) < .1, f"slope {s1:.3f}")
check("Theorem 2.2: sup |w_a - w_bar_a| = O(1/alpha^2) (slope ~ -2)", abs(s2 + 2) < .15, f"slope {s2:.3f}")
# notes' example: h(w) = w + w^2, alpha = 100, fit alpha h(w) = 1
wex = (-1 + np.sqrt(1.04)) / 2
check("notes' example: w = 0.00990195, linearized 0.01, gap 9.8e-5", abs(wex - .00990195) < 1e-8 and abs(.01 - wex - 9.8e-5) < 1e-6)

# ---- Fig. 1 (a)-(b) ----
R = np.random.default_rng(110)
n, m, d = 15, 20, 2
th = R.uniform(0, 2 * np.pi, n); X = np.c_[np.cos(th), np.sin(th)]
TB = R.normal(size=(3, d)); TB /= np.linalg.norm(TB, axis=1, keepdims=True); TA = R.choice([-1., 1.], 3)   # |a_j b_j| = 1
relu = lambda z: np.maximum(z, 0)
Y = relu(X @ TB.T) @ TA


def lossf(A, Bm):
    return .5 * np.mean((relu(X @ Bm.T) @ A - Y) ** 2)


def grads(A, Bm):
    Z = X @ Bm.T; r = relu(Z) @ A - Y
    return relu(Z).T @ r / n, ((r[:, None] * (Z > 0)) * A[None, :]).T @ X / n


A0 = R.normal(size=4); B0 = R.normal(size=(4, d))
check_grad("two-layer loss (a, b)", lambda p: lossf(p[:4], p[4:].reshape(4, d)), lambda p: np.concatenate([grads(p[:4], p[4:].reshape(4, d))[0], grads(p[:4], p[4:].reshape(4, d))[1].ravel()]), [np.r_[A0, B0.ravel()]])


def train(tau, steps):
    Rr = np.random.default_rng(7)
    a = Rr.normal(0, tau, m // 2); b = Rr.normal(0, tau, (m // 2, d))
    A, Bm = np.r_[a, -a], np.r_[b, b]                      # symmetrized: zero output at init
    Z0 = X @ Bm.T; J = np.c_[relu(Z0), ((Z0 > 0) * A[None, :])[:, :, None].repeat(d, 2).reshape(n, -1) * np.tile(X, m)]   # Jacobian of outputs
    eta = min(.2, 1 / np.linalg.eigvalsh(J.T @ J / n)[-1])  # step 1/lambda_max of the tangent kernel at init (capped for small tau)
    path = [np.c_[np.abs(A)[:, None] * Bm, np.sign(A)]]; L = [lossf(A, Bm)]; A0_, B0_ = A.copy(), Bm.copy()
    for t in range(steps):
        gA, gB = grads(A, Bm); A, Bm = A - eta * gA, Bm - eta * gB
        if (t + 1) % (steps // 120) == 0:
            path.append(np.c_[np.abs(A)[:, None] * Bm, np.sign(A)]); L.append(lossf(A, Bm))
    rel = np.linalg.norm(np.c_[A - A0_, Bm - B0_]) / np.linalg.norm(np.c_[A0_, B0_])
    return np.array(path), np.array(L), rel


out = {"alphas": alphas.tolist(), "dmax": sig(res[:, 0]), "emax": sig(res[:, 1]), "TB": r(TB, 4), "TA": TA.tolist()}
for tau, steps in ((.1, 60000), (5.0, 24000)):
    P, L, rel = train(tau, steps)
    print(f"  tau = {tau}: loss {L[0]:.3f} -> {L[-1]:.2e}, relative weight change {rel:.3f}")
    out[f"path{tau}"] = [r(p, 3) for p in P]; out[f"loss{tau}"] = sig(np.maximum(L, 1e-12), 3); out[f"rel{tau}"] = round(float(rel), 4)
check("tau = 5 (lazy): fits the data with a relative weight change < 0.15", out["loss5.0"][-1] < 1e-3 and out["rel5.0"] < .15)
check("tau = 0.1 (non-lazy): fits the data and the weights move far from init (relative change > 2)", out["loss0.1"][-1] < 1e-3 and out["rel0.1"] > 2)
scan = [(tau, train(tau, 24000)[2]) for tau in (.1, .25, .5, 1., 2., 5., 10.)]
sl = np.polyfit(np.log([t for t, _ in scan[3:]]), np.log([v for _, v in scan[3:]]), 1)[0]
check("lazy regime: relative weight change ~ tau^-2 for tau >= 1 (slope ~ -2)", abs(sl + 2) < .3, f"slope {sl:.2f}")
out["scan"] = [[t, round(float(v), 5)] for t, v in scan]
save_json("chizat2019lazy", "data.json", out)
done()
