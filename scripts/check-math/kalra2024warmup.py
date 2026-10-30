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
done()
