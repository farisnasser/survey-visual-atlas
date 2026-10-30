"""Cohen, Kaur, Li, Kolter, Talwalkar 2021: gradient descent typically occurs at the edge of stability.

1. Quadratic stability (Sec. 2, Eq. 1): GD diverges along an eigenvector with a > 2/eta; heavy ball diverges past
   MSS_Polyak = (2 + 2 beta)/eta, Nesterov past MSS_Nesterov = (2 + 2 beta)/(eta (1 + 2 beta)).
2. The paper's Fig. 2 quadratic (eigenvalues 20 and 1) for the 3D tab.
3. Offline replication: a small tanh MLP trained by full-batch GD on synthetic regression data, sharpness by power
   iteration with Hessian-vector products (finite differences of exact gradients). Written to data.json.
"""
from common import *

print("cohen2021gradient")

# ---------- 1. quadratic stability thresholds ----------
def rho_gd(a, eta):
    return abs(1 - eta * a)


def rho_mom(a, eta, beta, nesterov=False):
    # heavy ball: v' = beta v - eta a x ; x' = x + v'   (Cohen et al. App. B form)
    # Nesterov:   v' = beta v - eta a (x + beta v) ; x' = x + v'
    if nesterov:
        M = np.array([[beta - eta * a * beta, -eta * a], [beta - eta * a * beta, 1 - eta * a]])
    else:
        M = np.array([[beta, -eta * a], [beta, 1 - eta * a]])
    return max(abs(np.linalg.eigvals(M)))


eta = 0.1
check("GD on a quadratic: stable iff a < 2/eta", rho_gd(19.9, eta) < 1 and rho_gd(20.1, eta) > 1)
for beta in [0.5, 0.9]:
    mss = (2 + 2 * beta) / eta
    check(f"heavy-ball MSS (2+2b)/eta, beta = {beta}", rho_mom(mss * .999, eta, beta) < 1 and rho_mom(mss * 1.001, eta, beta) > 1, f"MSS = {mss:.1f}")
    mssn = (2 + 2 * beta) / (eta * (1 + 2 * beta))
    check(f"Nesterov MSS (2+2b)/(eta(1+2b)), beta = {beta}", rho_mom(mssn * .999, eta, beta, True) < 1 and rho_mom(mssn * 1.001, eta, beta, True) > 1, f"MSS = {mssn:.2f}")
# Fig. 2 of the paper: eigenvalues 20 and 1, eta = 0.11 -> 2/eta = 18.2 < 20: diverges along q1 only
check("Fig. 2 example: a1 = 20 > 2/0.11 > a2 = 1", rho_gd(20, .11) > 1 and rho_gd(1, .11) < 1, f"|1 - 0.11*20| = {rho_gd(20, .11):.2f}")

# ---------- 2. the 3D tab: Fig. 2 quadratic f = 1/2 (20 x^2 + y^2) ----------
f2 = lambda z: .5 * (20 * z[0] ** 2 + z[1] ** 2)
check_grad("Fig. 2 quadratic", f2, lambda z: np.array([20 * z[0], z[1]]), rand_pts(4, -1, 1))
for eta, expect in [(.09, "converge"), (.11, "diverge")]:
    z = np.array([.05, 1.5])
    for t in range(60):
        z = z - eta * np.array([20 * z[0], z[1]])
    ok = (abs(z[0]) < 1e-3) if expect == "converge" else (abs(z[0]) > 1)
    check(f"Fig. 2 quadratic, eta = {eta}: {expect}s along q1", ok, f"x after 60 steps = {z[0]:.2e}, y = {z[1]:.2e}")

# ---------- 3. offline MLP replication ----------
rng = np.random.default_rng(0)
N, D, W = 128, 10, 48
X = rng.normal(size=(N, D))
Y = np.sin(X @ rng.normal(size=D) / np.sqrt(D) * 2) + .3 * np.tanh(X @ rng.normal(size=D))


def init():
    r = np.random.default_rng(1)
    return [r.normal(size=(D, W)) / np.sqrt(D), np.zeros(W), r.normal(size=(W, W)) / np.sqrt(W), np.zeros(W), r.normal(size=(W, 1)) / np.sqrt(W), np.zeros(1)]


def flat(P):
    return np.concatenate([p.ravel() for p in P])


shapes = [p.shape for p in init()]


def unflat(v):
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s)); out.append(v[i:i + n].reshape(s)); i += n
    return out


def loss_grad(v):
    W1, b1, W2, b2, W3, b3 = unflat(v)
    h1 = np.tanh(X @ W1 + b1); h2 = np.tanh(h1 @ W2 + b2); f = (h2 @ W3 + b3)[:, 0]
    r = f - Y; Lv = .5 * np.mean(r * r)
    df = (r / N)[:, None]
    gW3 = h2.T @ df; gb3 = df.sum(0)
    dh2 = df @ W3.T * (1 - h2 ** 2)
    gW2 = h1.T @ dh2; gb2 = dh2.sum(0)
    dh1 = dh2 @ W2.T * (1 - h1 ** 2)
    gW1 = X.T @ dh1; gb1 = dh1.sum(0)
    return Lv, flat([gW1, gb1, gW2, gb2, gW3, gb3])


v0 = flat(init())
# finite-difference check on a few random coordinates
gv = loss_grad(v0)[1]
idx = rng.choice(v0.size, 12, replace=False); err = 0
for i in idx:
    e = np.zeros_like(v0); e[i] = 1e-6
    err = max(err, abs((loss_grad(v0 + e)[0] - loss_grad(v0 - e)[0]) / 2e-6 - gv[i]))
check("MLP backprop gradient vs finite differences", err < 1e-7, f"max err {err:.1e}")


def hvp(v, w, h=1e-4):
    return (loss_grad(v + h * w)[1] - loss_grad(v - h * w)[1]) / (2 * h)


def sharp(v, u, iters):
    for _ in range(iters):
        hu = hvp(v, u); lam = u @ hu; u = hu / np.linalg.norm(hu)
    return lam, u


ETAS = [.1, .3, .36, .42]
curves = {}
for eta in ETAS:
    v = v0.copy(); u = rng.normal(size=v.size); u /= np.linalg.norm(u)
    lam, u = sharp(v, u, 60)
    rows = []
    for t in range(3001):
        Lv, g = loss_grad(v)
        if t % 10 == 0:
            lam, u = sharp(v, u, 8 if t else 60)
            rows.append([t, Lv, lam])
        v = v - eta * g
    curves[str(eta)] = rows
    R = np.array(rows)
    print(f"    eta={eta}: S0={R[0, 2]:.2f} maxS={R[:, 2].max():.2f} lateS={R[-50:, 2].mean():.2f} 2/eta={2 / eta:.2f} final loss {R[-1, 1]:.2e}")
R = {e: np.array(curves[str(e)]) for e in ETAS}
check("MLP: progressive sharpening (sharpness grows from its initial value)", all(R[e][:, 2].max() > 1.4 * R[e][0, 2] for e in ETAS), f"S0 = {R[ETAS[0]][0, 2]:.2f}, max = {[round(R[e][:, 2].max(), 2) for e in ETAS]}")
check("MLP eta = 0.1: sharpness saturates below 2/eta (no EoS)", R[.1][:, 2].max() < 2 / .1)
for e in ETAS[1:]:
    late = R[e][-100:, 2].mean()
    check(f"MLP eta = {e}: late sharpness within 10% of 2/eta", abs(late - 2 / e) / (2 / e) < .1, f"{late:.2f} vs {2 / e:.2f}")
check("MLP: losses decrease overall", all(R[e][-1, 1] < .2 * R[e][0, 1] for e in ETAS))
spikes = {e: int((np.diff(R[e][:, 1]) > 0).sum()) for e in ETAS}
check("MLP: loss is non-monotone at large eta", all(spikes[e] > 0 for e in ETAS[1:]) and spikes[ETAS[0]] == 0, f"increasing 10-step windows: {spikes}")
save_json("cohen2021gradient", "data.json", {"etas": ETAS, "curves": {k: [[int(r[0]), float(f"{r[1]:.4g}"), round(float(r[2]), 3)] for r in v] for k, v in curves.items()},
                                             "setup": f"tanh MLP {D}-{W}-{W}-1, {N} points, MSE, full-batch GD"})
done()
