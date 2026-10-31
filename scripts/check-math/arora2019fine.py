"""Arora, Du, Hu, Li, Wang 2019: fine-grained (spectral) convergence and the width-free complexity y^T (H_inf)^-1 y.

Network (Sec. 3): f(x) = (1/sqrt m) sum_r a_r relu(w_r . x), a_r = +-1 fixed, w_r ~ N(0, kappa^2 I), loss 1/2 |y - u|^2.
H_inf_ij = E_w[x_i.x_j 1{w.x_i >= 0, w.x_j >= 0}] = x_i.x_j (pi - arccos(x_i.x_j)) / (2 pi) for unit x (Eq. 6).
Theorem 4.1: |y - u(k)| ~ |(I - eta H_inf)^k y| = sqrt(sum_i (1 - eta lambda_i)^{2k} (v_i.y)^2) (kappa small so u(0) ~ 0;
with kappa = 1 we keep the initial output and compare with |(I - eta H_inf)^k (y - u(0))|).
Theorem 5.1: generalization gap of the trained net <= sqrt(2 y^T H_inf^-1 y / n) + small terms (1-Lipschitz loss).
Data: n = 60 unit vectors in R^10. Labels: "structured" y = sign(x_1) (a simple, learnable rule) vs random +-1.
"""
from common import *

print("arora2019fine")
n, d, m, kappa = 60, 10, 8000, 1.0
R0 = np.random.default_rng(87)
X = R0.normal(size=(n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
G = np.clip(X @ X.T, -1, 1)
Hinf = G * (np.pi - np.arccos(G)) / (2 * np.pi)
lam, V = np.linalg.eigh(Hinf)
check("H_inf is positive definite (lambda_0 > 0)", lam[0] > 0, f"lambda_min = {lam[0]:.4f}, lambda_max = {lam[-1]:.3f}")
# Monte Carlo check of the closed form
W = R0.normal(size=(200000, d))
S = (W @ X[:6].T >= 0).astype(float)
Hmc = (S.T @ S / len(W)) * G[:6, :6]
check("closed form of H_inf matches Monte Carlo", np.max(np.abs(Hmc - Hinf[:6, :6])) < 4e-3, f"max err {np.max(np.abs(Hmc - Hinf[:6, :6])):.1e}")
y_s = np.sign(X[:, 0]); y_r = R0.choice([-1., 1.], n)
cost = lambda y: float(y @ np.linalg.solve(Hinf, y))
proj = lambda y: (V.T @ y) ** 2
print(f"  y^T H^-1 y: structured {cost(y_s):.1f}, random {cost(y_r):.1f}")
check("structured labels have a much smaller y^T H^-1 y than random labels", cost(y_s) < .5 * cost(y_r))
top = slice(n - 10, n)
check("structured labels put more mass on the top-10 eigenvectors", proj(y_s)[top].sum() / n > proj(y_r)[top].sum() / n, f"{proj(y_s)[top].sum() / n:.2f} vs {proj(y_r)[top].sum() / n:.2f}")
bound = lambda y: np.sqrt(2 * cost(y) / n)
print(f"  Theorem 5.1 leading term sqrt(2 y^T H^-1 y / n): structured {bound(y_s):.2f}, random {bound(y_r):.2f}")

relu = lambda z: np.maximum(z, 0)
Rw = np.random.default_rng(1)
W0 = Rw.normal(0, kappa, (m, d)); a = Rw.choice([-1., 1.], m)
u_of = lambda W: relu(X @ W.T) @ a / np.sqrt(m)


def grad(W, y):
    Z = X @ W.T; r = relu(Z) @ a / np.sqrt(m) - y
    return ((r[:, None] * (Z >= 0)) * a[None, :]).T @ X / np.sqrt(m)


ms, Ws_, as_ = 12, Rw.normal(0, 1, (12, d)), Rw.choice([-1., 1.], 12)
us = lambda w: relu(X @ w.reshape(ms, d).T) @ as_ / np.sqrt(ms)
gs = lambda w: (((relu(X @ w.reshape(ms, d).T) @ as_ / np.sqrt(ms) - y_s)[:, None] * (X @ w.reshape(ms, d).T >= 0)) * as_[None, :]).T @ X / np.sqrt(ms)
check_grad("Phi(W) (Eq. 5), m = 12", lambda w: .5 * np.sum((us(w) - y_s) ** 2), lambda w: gs(w).ravel(), [Ws_.ravel(), .5 * Ws_.ravel()], tol=1e-4)
eta = 1 / lam[-1]
K = 300


def run(y):
    W = W0.copy(); L = [.5 * np.sum((u_of(W) - y) ** 2)]
    for k in range(K):
        W = W - eta * grad(W, y); L.append(.5 * np.sum((u_of(W) - y) ** 2))
    return np.array(L), W


u0 = u_of(W0)
pred = lambda y: np.array([.5 * np.sum((1 - eta * lam) ** (2 * k) * proj(y - u0)) for k in range(K + 1)])
Ls, Ws = run(y_s); Lr, Wr = run(y_r)
Ps, Pr = pred(y_s), pred(y_r)
err = max(np.max(np.abs(np.sqrt(2 * Ls) - np.sqrt(2 * Ps))), np.max(np.abs(np.sqrt(2 * Lr) - np.sqrt(2 * Pr)))) / np.sqrt(n)
check("Theorem 4.1: |y - u(k)| tracks |(I - eta H_inf)^k (y - u(0))| (max gap / sqrt n < 0.03)", err < .03, f"{err:.4f}")
check("structured labels train faster than random labels", Ls[100] < .5 * Lr[100], f"loss at k = 100: {Ls[100]:.3f} vs {Lr[100]:.3f}")
mv = lambda W: np.linalg.norm(W - W0, axis=1).max()
print(f"  max neuron movement / kappa: structured {mv(Ws) / kappa:.3f}, random {mv(Wr) / kappa:.3f}")
# weight-space slice for the 3-D tab: W0 + alpha * D_top + beta * D_bot, D = Z q (Z = Jacobian of u at W0 in direction of eigvec q)
Z0 = (X @ W0.T >= 0)


def dirW(q):   # weight perturbation whose first-order effect on u is H q (normalized to unit effect on u along q)
    Dw = ((q[:, None] * Z0) * a[None, :]).T @ X / np.sqrt(m)
    return Dw


Dt, Db = dirW(V[:, -1]), dirW(V[:, 0])
Dt /= np.linalg.norm(Dt); Db /= np.linalg.norm(Db)
yv = y_s
c = np.linalg.norm(u_of(W0 + 1e-3 * Dt) - u_of(W0)) / 1e-3
A = np.linspace(-3, 3, 31) / c * np.sqrt(n) * .25; B = np.linspace(-3, 3, 31) / c * np.sqrt(n) * .25 * 3
grid = [[.5 * np.sum((u_of(W0 + al * Dt + be * Db) - yv) ** 2) for al in A] for be in B]
ga = [float(A[0]), float(A[-1]), float(B[0]), float(B[-1])]
print(f"  slice span alpha in [{ga[0]:.2f}, {ga[1]:.2f}], beta in [{ga[2]:.2f}, {ga[3]:.2f}]")
save_json("arora2019fine", "data.json", {"lam": sig(lam, 4), "ps": sig(proj(y_s), 4), "pr": sig(proj(y_r), 4), "Ls": sig(Ls, 4), "Lr": sig(Lr, 4), "Ps": sig(Ps, 4), "Pr": sig(Pr, 4),
                                       "eta": float(f"{eta:.5g}"), "cs": round(cost(y_s), 3), "cr": round(cost(y_r), 3), "n": n,
                                       "slice": [sig(row, 4) for row in grid], "dom": ga})
# in-page notes (papers/arora2019fine/notes.md): H = diag(2, 0.2), eta = 0.25, y = (1, 1), u0 = 0
Hn = np.diag([2., .2]); yn = np.array([1., 1.]); un = np.zeros(2); es = []
for _ in range(3):
    un = un - .25 * Hn @ (un - yn); es.append(un - yn)
check("notes example: multipliers 0.5 and 0.95", np.allclose(np.diag(np.eye(2) - .25 * Hn), [.5, .95]))
check("notes example: e1 = (-0.5, -0.95), e2 = (-0.25, -0.9025)", np.allclose(es[0], [-.5, -.95]) and np.allclose(es[1], [-.25, -.9025]))
check("notes example: ||e_t||^2 = 0.5^(2t) + 0.95^(2t)", all(np.isclose(es[t - 1] @ es[t - 1], .5 ** (2 * t) + .95 ** (2 * t)) for t in (1, 2, 3)))
check("notes: y^T H^-1 y = 5.5 for y = (1, 1), 0.5 for y = (1, 0)", np.isclose(yn @ np.linalg.solve(Hn, yn), 5.5) and np.isclose(np.array([1, 0]) @ np.linalg.solve(Hn, [1, 0]), .5))
# primer idea 1: unit inputs at 60 deg: H12 = 0.5 (pi - pi/3)/(2 pi) = 1/6, H11 = 1/2; Monte Carlo of the indicator definition
x1p, x2p = np.array([1., 0.]), np.array([.5, np.sqrt(3) / 2])
hcf = lambda a, b: a @ b * (np.pi - np.arccos(np.clip(a @ b, -1, 1))) / (2 * np.pi)
Rw = np.random.default_rng(9).normal(size=(400000, 2))
mc12 = x1p @ x2p * np.mean((Rw @ x1p >= 0) & (Rw @ x2p >= 0))
check("primer: H12 = 1/6 (closed form and Monte Carlo), H11 = 1/2", np.isclose(hcf(x1p, x2p), 1 / 6) and abs(mc12 - 1 / 6) < 3e-3 and np.isclose(hcf(x1p, x1p), .5), f"MC {mc12:.4f}")
Hp = np.array([[.5, 1 / 6], [1 / 6, .5]]); wv, Vv = np.linalg.eigh(Hp)
check("primer: eigenvalues 1/3 along (1,-1)/sqrt2 and 2/3 along (1,1)/sqrt2", np.allclose(wv, [1 / 3, 2 / 3]) and np.isclose(abs(Vv[:, 1] @ [1, 1]) / np.sqrt(2), 1) and np.isclose(abs(Vv[:, 0] @ [1, -1]) / np.sqrt(2), 1))
# primer idea 3: eta = 1: residual of y = (1,1) shrinks by 1/3 per step, of y = (1,-1) by 2/3
for yy, fac in [((1., 1.), 1 / 3), ((1., -1.), 2 / 3)]:
    e0 = -np.array(yy); e1 = (np.eye(2) - Hp) @ e0
    assert np.allclose(e1, fac * e0)
check("primer: with eta = 1 the residual shrinks by 1/3 (y = (1,1)) and by 2/3 (y = (1,-1)) per step", True)
# primer idea 4: y^T H^-1 y = 3 for (1, 1) and 6 for (1, -1)
check("primer: y^T H^-1 y = 3 and 6", np.isclose(np.array([1, 1]) @ np.linalg.solve(Hp, [1, 1]), 3) and np.isclose(np.array([1, -1]) @ np.linalg.solve(Hp, [1, -1]), 6))
done()
