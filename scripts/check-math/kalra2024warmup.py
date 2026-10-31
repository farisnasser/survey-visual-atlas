"""Kalra & Barkeshli 2024: why warmup the learning rate.

1. 3D tab: n = 1 linear model f = uv on two examples x = 1 with labels +-0.02 (SGD, one example per step), linear
   warmup eta_t = eta_trgt min(1, t/T_wrm) vs constant eta_trgt: warmup lets SGD tolerate eta_trgt lambda_0 > 4.
   (Deterministic GD from a converged point cannot catapult: the residual is ~0, so the label noise matters.)
2. Adam preconditioner P_t = (1 - beta1^t) diag(sqrt(v_hat_t)) + eps (Eq. 9); preconditioned sharpness lambda_max(P^-1 H).
   GI-Adam (v_0 = g_0^2) lowers it at the first step by a factor sqrt(1 - beta2).
"""
from common import *

print("kalra2024warmup")

# ---------- 1. warmup on the n = 1 model, SGD on two examples with labels +-SIG ----------
SIG = .02
L = lambda z: .5 * (z[0] * z[1]) ** 2 + .5 * SIG ** 2          # full loss = mean of the two per-example losses
G = lambda z: np.array([z[0] * z[1] ** 2, z[1] * z[0] ** 2])
check_grad("full loss (uv)^2/2 + sig^2/2", L, G, rand_pts(5, -2, 2))
U0 = np.array([2.0, .35]); LAM0 = U0 @ U0


def sgd(c, Tw, seed, T=1000):
    """eta_t = (c / lambda0) min(1, t/Tw); one example per step (label +SIG or -SIG); mulberry32 is used in the page, so
    here we only check statistics over seeds."""
    r = np.random.default_rng(seed); z = U0.copy(); ET = c / LAM0; out = []
    for t in range(T):
        eta = ET * min(1, (t + 1) / Tw); yi = SIG if r.random() < .5 else -SIG
        z = z - eta * (z[0] * z[1] - yi) * np.array([z[1], z[0]])
        if not np.all(np.isfinite(z)) or np.abs(z).max() > 30:
            return np.array(out), False
        out.append((t + 1, eta, z @ z, L(z)))
    return np.array(out), True


C0 = 6
wins = {Tw: sum(sgd(C0, Tw, s)[1] for s in range(20)) for Tw in [1, 20, 50, 200]}
check("eta_trgt lambda0 = 6 without warmup diverges (20/20 seeds)", wins[1] == 0, f"successes out of 20: {wins}")
check("linear warmup of >= 50 steps trains at the same eta_trgt (20/20)", wins[50] == 20 and wins[200] == 20)
o, ok = sgd(C0, 200, 0)
check("T_wrm = 200: final sharpness below 2/eta_trgt", ok and o[-100:, 2].mean() < 2 / (C0 / LAM0), f"lambda = {o[-100:, 2].mean():.3f} < {2 / (C0 / LAM0):.3f}")
first = int(np.argmax(o[:, 1] * o[:, 2] > 2))
trk = o[first + 5:200]
check("during warmup the sharpness tracks 2/eta_t once it is first reached", np.all(trk[:, 1] * trk[:, 2] < 4), f"first eta*lambda > 2 at step {first + 1}; max eta*lambda afterwards {(trk[:, 1] * trk[:, 2]).max():.2f}")
# success map for the sweep tab
CS_, TWS = [2, 3, 4, 5, 6, 8, 10, 12], [1, 5, 10, 20, 50, 100, 200, 400]
grid = [[sum(sgd(c, Tw, s)[1] for s in range(10)) for Tw in TWS] for c in CS_]
check("no warmup works only below eta lambda0 = 4", all(grid[k][0] == 10 for k, c in enumerate(CS_) if c < 4) and all(grid[k][0] == 0 for k, c in enumerate(CS_) if c > 4))
check("longer warmup tolerates larger eta_trgt", grid[CS_.index(12)][-1] > grid[CS_.index(12)][2], f"eta lambda0 = 12: {grid[CS_.index(12)]}")
DATA = {"cs": CS_, "tws": TWS, "grid": grid}

# ---------- 3. Adam vs GI-Adam preconditioned sharpness ----------
rng = np.random.default_rng(0)
d = 200
h = np.exp(rng.uniform(np.log(.1), np.log(10), d))      # quadratic L = 1/2 sum h_i th_i^2
th0 = rng.normal(size=d) * np.exp(rng.uniform(np.log(1e-4), 0, d))   # some gradient components near zero
b1, b2, eps, eta = .9, .999, 1e-8, 1e-3


def adam(gi, T=300):
    th = th0.copy(); m = np.zeros(d); g0 = h * th; v = g0 ** 2 if gi else np.zeros(d); out = []
    for t in range(1, T + 1):
        g = h * th
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        th = th - eta * mh / (np.sqrt(vh) + eps)
        P = (1 - b1 ** t) * np.sqrt(vh) + eps
        out.append((h / P).max())
    return np.array(out)


A, GI = adam(False), adam(True)
ratio = GI[0] / A[0]
check("GI-Adam lowers the first-step preconditioned sharpness by ~sqrt(1 - beta2)", abs(ratio - np.sqrt(1 - b2)) / np.sqrt(1 - b2) < .02, f"ratio {ratio:.4f} vs sqrt(1 - b2) = {np.sqrt(1 - b2):.4f}; Adam {A[0]:.2e}, GI {GI[0]:.2e}")
check("standard Adam starts with a huge preconditioned sharpness", A[0] > 1e4 * h.max(), f"{A[0]:.2e} vs lambda_max(H) = {h.max():.2f}")
check("GI-Adam's effective step: eta sqrt(1 - beta2^t) (automatic warmup)", abs(np.sqrt(1 - b2 ** 1) - .0316) < 1e-3)
DATA.update({"adam": sig(A, 4), "gi": sig(GI, 4), "thr": round((2 + 2 * b1) / (1 - b1) / eta, 1)})
save_json("kalra2024warmup", "data.json", DATA)
# notes examples
check("notes: P^{1/2} H P^{1/2} with H = diag(20, 1), P = diag(0.01, 1) has lambda_max 1", abs(eig(np.diag([.2, 1.]))[-1] - 1) < 1e-12)
check("notes: 2/20 = 0.1 < 0.3 < 2/5 = 0.4", 2 / 20 < .3 < 2 / 5)
# in-page notes (papers/kalra2024warmup/notes.md): Sec. 2 thresholds 2/20 = 0.1, 2/5 = 0.4; Sec. 5 P^1/2 H P^1/2
check("notes: lambda = 20 -> eta < 0.1 (0.3 unstable); lambda = 5 -> 0.4 (0.3 stable)", np.isclose(2 / 20, .1) and abs(1 - .3 * 20) > 1 and np.isclose(2 / 5, .4) and abs(1 - .3 * 5) < 1)
Pk, Hk = np.diag([.01, 1.]), np.diag([20., 1.])
Mk = np.sqrt(Pk) @ Hk @ np.sqrt(Pk)
check("notes: P^1/2 H P^1/2 = diag(0.2, 1), largest eigenvalue 1 (raw 20)", np.allclose(Mk, np.diag([.2, 1.])) and np.isclose(eig(Mk)[-1], 1) and np.isclose(eig(Hk)[-1], 20))
check("notes: P^1/2 H P^1/2 has the eigenvalues of P H (the preconditioned update's Jacobian)", np.allclose(np.sort(np.linalg.eigvals(Pk @ Hk).real), eig(Mk)))
# primer: linear warmup value, 2/eta range, catapult in L = (uv - 1)^2 / 2 with eta = 0.6, Adam / GI-Adam first step
eta_t = lambda t, a=.01, b=.3, T=100: a + (b - a) * min(t, T) / T
check("primer: eta_50 = 0.155; 2/eta from 200 to ~6.7", np.isclose(eta_t(50), .155) and np.isclose(2 / eta_t(0), 200) and np.isclose(2 / eta_t(100), 6.67, atol=.01))
Luv = lambda p: .5 * (p[0] * p[1] - 1) ** 2
gu = lambda p: (p[0] * p[1] - 1) * np.array([p[1], p[0]])
for a in [.5, 1., 2.2]:
    assert np.isclose(eig(fd_hess(Luv, np.array([a, 1 / a])))[-1], a * a + 1 / a ** 2, atol=1e-3)
check("primer: sharpness at the minimum (a, 1/a) is a^2 + 1/a^2, smallest (2) at (1, 1)", True)
p = np.array([2.2, .45]); l0, s0 = Luv(p), eig(fd_hess(Luv, p))[-1]; mx = 0
for t in range(3000):
    p = p - .6 * gu(p); mx = max(mx, Luv(p))
s1 = eig(fd_hess(Luv, p))[-1]
check("primer: catapult from (2.2, 0.45), eta = 0.6: loss 0.00005, sharpness ~5.0 > 2/eta ~3.3, spike ~1.5, ends at sharpness ~2.0",
      np.isclose(l0, 5e-5, rtol=.01) and np.isclose(s0, 5.0, atol=.05) and s0 > 2 / .6 and np.isclose(2 / .6, 3.3, atol=.05)
      and np.isclose(mx, 1.5, atol=.06) and np.isclose(s1, 2.0, atol=.02) and Luv(p) < 1e-20, f"loss0 {l0:.2e}, S0 {s0:.3f}, peak {mx:.3f}, S_end {s1:.4f}")
b1, b2, lr = .9, .999, 1e-3
g0 = np.array([10., .001])
def adam_first(v0):
    m = (1 - b1) * g0; v = b2 * v0 + (1 - b2) * g0 ** 2
    return lr * (m / (1 - b1)) / np.sqrt(v / (1 - b2))
s_adam, s_gi = adam_first(np.zeros(2)), adam_first(g0 ** 2)
check("primer: Adam's first step is eta sign(g0) for gradients 10 and 0.001", np.allclose(s_adam, lr * np.sign(g0)))
check("primer: GI-Adam (v0 = g0^2) first step is sqrt(1 - beta2) = 0.0316 times Adam's", np.allclose(s_gi / s_adam, np.sqrt(1 - b2)) and np.isclose(np.sqrt(1 - b2), .0316, atol=5e-5))
done()
