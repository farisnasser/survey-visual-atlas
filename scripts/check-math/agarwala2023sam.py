"""Agarwala & Dauphin 2023: SAM operates far from home (eigenvalue regularization as a dynamical phenomenon).

Unnormalized SAM, Eq. (6): theta_{t+1} = theta_t - eta grad L(theta_t + rho grad L(theta_t)).
On 1/2 x^T H x, Eq. (12): x_{t+1} - x_t = -eta (H + rho H^2) x_t, stable iff eta (lam_max + rho lam_max^2) < 2:
the SAM edge of stability, lam_SAM = (-1 + sqrt(1 + 8 rho / eta)) / (2 rho) < 2/eta.
1. 3D tab: n = 1 linear model L(u, v) = (uv - 1)^2 / 2 (minima uv = 1 with sharpness u^2 + v^2): GD and SAM from the same
   unbalanced start; SAM must settle on a flatter minimum.
2. Offline replication: tanh MLP trained by full-batch GD and by unnormalized SAM; top Hessian eigenvalue vs 2/eta and lam_SAM.
"""
from common import *

print("agarwala2023sam")
lam_sam = lambda eta, rho: (-1 + np.sqrt(1 + 8 * rho / eta)) / (2 * rho)
# notes example: eta = 0.5, rho = 0.2
check("notes: lam_SAM(0.5, 0.2) = 2.62 < 2/eta = 4", abs(lam_sam(.5, .2) - 2.6235) < 1e-3, f"{lam_sam(.5, .2):.4f}")
check("notes: at lam = 3, GD multiplier -0.5, SAM multiplier -1.4", abs(1 - .5 * 3 + .5) < 1e-12 and abs(1 - .5 * (3 + .2 * 9) + 1.4) < 1e-12)
# Eq. (12) on a 2x2 quadratic: SAM step equals the (H + rho H^2) linear map
H = np.array([[3., .4], [.4, 1.]]); x = np.array([.7, -.3]); eta, rho = .3, .1
g = lambda z: H @ z
x1 = x - eta * g(x + rho * g(x))
check("Eq. (12): unnormalized SAM on a quadratic = x - eta (H + rho H^2) x", np.allclose(x1, x - eta * (H + rho * H @ H) @ x))
for e_, r_ in [(.3, .1), (.1, .05)]:
    ls = lam_sam(e_, r_)
    check(f"SAM edge eta = {e_}, rho = {r_}: lam = {ls:.3f} solves eta(lam + rho lam^2) = 2", abs(e_ * (ls + r_ * ls * ls) - 2) < 1e-12 and ls < 2 / e_)

# ---------- 1. UV model ----------
L = lambda z: .5 * (z[0] * z[1] - 1) ** 2
G = lambda z: (z[0] * z[1] - 1) * np.array([z[1], z[0]])
check_grad("(uv - 1)^2/2", L, G, rand_pts(5, -2, 2))
check_hess_at("on the minimum (2, 0.5): top eigenvalue u^2 + v^2 = 4.25", L, [2, .5], [[.25, 1], [1, 4]])
Z0 = np.array([2.2, .1])


def run(eta, rho, T=6000):
    z = Z0.copy()
    for _ in range(T):
        z = z - eta * G(z + rho * G(z))
        if np.abs(z).max() > 1e3: return z, False
    return z, True


ETA3 = .3
ends = {}
for rho in [0, .05, .1, .2]:
    z, ok = run(ETA3, rho)
    lam = z @ z; ends[rho] = lam
    thr = 2 / ETA3 if rho == 0 else lam_sam(ETA3, rho)
    print(f"    eta {ETA3} rho {rho}: end {np.round(z, 4)}, residual {z[0] * z[1] - 1:.1e}, sharpness {lam:.4f}, threshold {thr:.4f}")
    check(f"UV, rho = {rho}: converges to a minimum with sharpness below its threshold", ok and abs(z[0] * z[1] - 1) < 1e-8 and lam < thr + 1e-9)
check("larger rho ends at a flatter minimum (same start, same eta)", ends[0] > ends[.05] > ends[.1] > ends[.2], f"{ {k: round(v, 3) for k, v in ends.items()} }")

# ---------- 2. MLP replication ----------
rng = np.random.default_rng(0)
N, D, W = 128, 10, 48
X = rng.normal(size=(N, D))
Y = np.sin(X @ rng.normal(size=D) / np.sqrt(D) * 2) + .3 * np.tanh(X @ rng.normal(size=D))
r1 = np.random.default_rng(1)
P0 = [r1.normal(size=(D, W)) / np.sqrt(D), np.zeros(W), r1.normal(size=(W, W)) / np.sqrt(W), np.zeros(W), r1.normal(size=(W, 1)) / np.sqrt(W), np.zeros(1)]
shapes = [p.shape for p in P0]
v0 = np.concatenate([p.ravel() for p in P0])


def unflat(v):
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s)); out.append(v[i:i + n].reshape(s)); i += n
    return out


def loss_grad(v):
    W1, b1, W2, b2, W3, b3 = unflat(v)
    h1 = np.tanh(X @ W1 + b1); h2 = np.tanh(h1 @ W2 + b2); f = (h2 @ W3 + b3)[:, 0]
    r = f - Y; df = (r / N)[:, None]
    dh2 = df @ W3.T * (1 - h2 ** 2); dh1 = dh2 @ W2.T * (1 - h1 ** 2)
    return .5 * np.mean(r * r), np.concatenate([(X.T @ dh1).ravel(), dh1.sum(0), (h1.T @ dh2).ravel(), dh2.sum(0), (h2.T @ df).ravel(), df.sum(0)])


def sharp(v, u, iters):
    lam = 0
    for _ in range(iters):
        hu = (loss_grad(v + 1e-4 * u)[1] - loss_grad(v - 1e-4 * u)[1]) / 2e-4; lam = u @ hu; u = hu / np.linalg.norm(hu)
    return lam, u


ETA, RHOS = .3, [0, .05, .1]
curves = {}
for rho in RHOS:
    v = v0.copy(); u = rng.normal(size=v.size); u /= np.linalg.norm(u); lam, u = sharp(v, u, 60); rows = []
    for t in range(3001):
        Lv, gv = loss_grad(v)
        if t % 10 == 0:
            lam, u = sharp(v, u, 8 if t else 60); rows.append([t, float(f"{Lv:.4g}"), round(float(lam), 3)])
        if rho:
            gv = loss_grad(v + rho * gv)[1]
        v = v - ETA * gv
    curves[str(rho)] = rows
    R = np.array(rows); late = R[-100:, 2].mean(); thr = 2 / ETA if rho == 0 else lam_sam(ETA, rho)
    print(f"    MLP eta {ETA} rho {rho}: S0 {R[0, 2]:.2f}, max {R[:, 2].max():.2f}, late {late:.3f}, threshold {thr:.3f}, final loss {R[-1, 1]:.2e}")
    check(f"MLP rho = {rho}: late sharpness within 5% of {'2/eta' if rho == 0 else 'lam_SAM'}", abs(late - thr) / thr < .05, f"{late:.3f} vs {thr:.3f}")
# in-page notes (papers/agarwala2023sam/notes.md): primer numbers on L = lam x^2 / 2, lam = 3, eta = 0.5, rho = 0.2
lam_, eta_, rho_ = 3., .5, .2
Lq = lambda x: .5 * lam_ * x * x
check("primer 1: normalized perturbation to x = 1.2: loss 2.16, gradient 3.6", abs(Lq(1.2) - 2.16) < 1e-12 and abs(lam_ * 1.2 - 3.6) < 1e-12)
check("primer 1: unnormalized perturbation rho g = 0.6 -> x = 1.6, gradient 4.8 = (lam + rho lam^2) x", abs(lam_ * (1 + rho_ * lam_) - 4.8) < 1e-12 and abs(lam_ * 1.6 - 4.8) < 1e-12)
# primer 3: residual dynamics of unnormalized SAM on a linear model z = J th - y equal I - eta JJ^T (I + rho JJ^T)
rng_ = np.random.default_rng(4)
J_ = rng_.normal(size=(5, 8)) / 3; y_ = rng_.normal(size=5); th_ = rng_.normal(size=8)
z0 = J_ @ th_ - y_
th1 = th_ - eta_ * J_.T @ (J_ @ (th_ + rho_ * J_.T @ z0) - y_)
K_ = J_ @ J_.T
check("primer 3: SAM residual map z -> (I - eta JJ^T (I + rho JJ^T)) z (linear model)", np.allclose(J_ @ th1 - y_, (np.eye(5) - eta_ * K_ @ (np.eye(5) + rho_ * K_)) @ z0))
x = .05; xs_ = [x]
for _ in range(80):
    x = x - eta_ * lam_ * (x + rho_ * np.sign(x)); xs_.append(x)
check("primer 4: normalized SAM from 0.05 -> -0.325, then bounces at +-0.6", abs(xs_[1] + .325) < 1e-12 and abs(abs(xs_[-1]) - .6) < 1e-9 and abs(xs_[-1] + xs_[-2]) < 1e-9, f"{xs_[-2]:.6f}, {xs_[-1]:.6f}")
save_json("agarwala2023sam", "data.json", {"eta": ETA, "curves": curves})
done()
