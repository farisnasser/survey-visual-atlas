"""Nguyen 2021: global convergence of GD for deep ReLU nets by tracking only the last hidden layer (Theorem 2.2).

Network (row convention of the paper, Eq. 1): F_0 = X (N x n0), F_l = relu(F_{l-1} W_l) for l < L, F_L = F_{L-1} W_L,
Phi = 1/2 |F_L - Y|_F^2, all layers trained by GD.  Here L = 3, n0 = 4, n1 = 8, n2 = N = 3 (last hidden width = N,
"linear width"), one output.  Section 3.1 initialization: W_3 = 0 and (W_1, W_2) = beta * Gaussian, beta large enough that
conditions (7)-(9) hold with C_l = 1; step size from (10).  Theorem 2.2 then guarantees Phi_k <= (1 - eta alpha_0^2/8)^k Phi_0
with alpha_0 = sigma_min(F_{L-1}) at init; the proof keeps sigma_min(F_{L-1}) >= alpha_0/2 along the run.
"""
from common import *

print("nguyen2021proof")
N, n0, n1, n2 = 3, 4, 8, 3
R0 = np.random.default_rng(88)
X = R0.normal(size=(N, n0))
Y = np.array([[1.0], [-.5], [.8]])
relu = lambda z: np.maximum(z, 0)


def fwd(W1, W2, W3):
    F1 = relu(X @ W1); F2 = relu(F1 @ W2); return F1, F2, F2 @ W3


def loss(W1, W2, W3):
    return .5 * np.sum((fwd(W1, W2, W3)[2] - Y) ** 2)


def grads(W1, W2, W3):
    F1, F2, out = fwd(W1, W2, W3); E = out - Y
    g3 = F2.T @ E
    D2 = (E @ W3.T) * (F1 @ W2 > 0); g2 = F1.T @ D2
    D1 = (D2 @ W2.T) * (X @ W1 > 0); g1 = X.T @ D1
    return g1, g2, g3


shapes = [(n0, n1), (n1, n2), (n2, 1)]
split = lambda p: [p[a:b].reshape(s) for (a, b), s in zip([(0, 32), (32, 56), (56, 59)], shapes)]
p0 = R0.normal(size=59)
check_grad("Phi(W1, W2, W3) by backprop", lambda p: loss(*split(p)), lambda p: np.concatenate([g.ravel() for g in grads(*split(p))]), [p0, .5 * p0])

while True:   # base directions with sigma_min(F_{L-1}) > 0 (needed by Section 3.1); resample dead-unit draws
    G1, G2 = R0.normal(size=(n0, n1)) / np.sqrt(n0), R0.normal(size=(n1, n2)) / np.sqrt(n1)
    if np.linalg.svd(relu(relu(X @ G1) @ G2), compute_uv=False)[-1] > .05:
        break
Xf = np.linalg.norm(X)


def conds(beta):
    W1, W2, W3 = beta * G1, beta * G2, np.zeros((n2, 1))
    F2 = fwd(W1, W2, W3)[1]
    a0 = np.linalg.svd(F2, compute_uv=False)[-1]
    lb = [np.linalg.norm(W1, 2) + 1, np.linalg.norm(W2, 2) + 1, np.linalg.norm(W3, 2) + 1]   # lambda_bar_l with C_l = 1
    l1L, l1Lm = lb[0] * lb[1] * lb[2], lb[0] * lb[1]
    ph = np.sqrt(2 * loss(W1, W2, W3))
    c7 = a0 ** 2 >= 16 * Xf * max(l1L / lb[l] / 1 for l in range(3)) * ph
    s = sum(l1Lm ** 2 / lb[l] ** 2 for l in range(2))
    c8 = a0 ** 3 >= 32 * Xf ** 2 * lb[2] * s * ph
    c9 = a0 ** 2 >= 16 * Xf ** 2 * lb[2] ** 2 * s
    eta = min(8 / a0 ** 2, Xf ** -2 * l1L ** -2 * sum(lb[l] ** -2 for l in range(2)) * sum(lb[l] ** -2 for l in range(3)) ** -2)
    return c7 and c8 and c9, a0, eta, (W1, W2, W3)


beta = 1.0
while not conds(beta)[0] and beta < 1e4:
    beta *= 1.25
ok, a0, eta_max, (W1, W2, W3) = conds(beta)
print(f"  smallest beta (x1.25 grid) meeting (7)-(9): {beta:.2f}; alpha_0 = {a0:.3f}; eta bound (10) = {eta_max:.3e}")
check("conditions (7), (8), (9) hold at the scaled initialization (Section 3.1, C_l = 1)", ok)
check("Section 3.1: W_L = 0 so sqrt(2 Phi_0) = |Y|_F", abs(np.sqrt(2 * loss(W1, W2, W3)) - np.linalg.norm(Y)) < 1e-12)
eta = .99 * eta_max
K = 4000
L, A, out = [loss(W1, W2, W3)], [a0], []
P = [W1, W2, W3]
for k in range(K):
    g = grads(*P); P = [p - eta * gg for p, gg in zip(P, g)]
    L.append(loss(*P)); A.append(np.linalg.svd(fwd(*P)[1], compute_uv=False)[-1])
    if k % 40 == 0:
        out.append(r(fwd(*P)[1], 4))
L, A = np.array(L), np.array(A)
bound = L[0] * (1 - eta * a0 ** 2 / 8) ** np.arange(K + 1)
check("Theorem 2.2, Eq. (11): Phi_k <= (1 - eta alpha_0^2/8)^k Phi_0 at every step", np.all(L <= bound * (1 + 1e-9)), f"final {L[-1]:.2e} vs bound {bound[-1]:.2e}")
check("proof invariant: sigma_min(F_{L-1}) stays >= alpha_0/2", A.min() >= a0 / 2, f"min alpha_t / alpha_0 = {A.min() / a0:.3f}")
# ordinary run (outside the theorem): same architecture and base directions, beta = 1, W_3 = 0, a normal step size
Pp = [G1.copy(), G2.copy(), np.zeros((n2, 1))]; a0p = np.linalg.svd(fwd(*Pp)[1], compute_uv=False)[-1]
etap = .5 / np.linalg.norm(fwd(*Pp)[1], 2) ** 2
Lp, Ap, Fp = [loss(*Pp)], [a0p], []
for k in range(K):
    g = grads(*Pp); Pp = [p - etap * gg for p, gg in zip(Pp, g)]
    Lp.append(loss(*Pp)); Ap.append(np.linalg.svd(fwd(*Pp)[1], compute_uv=False)[-1])
    if k % 20 == 0:
        Fp.append(r(fwd(*Pp)[1], 4))
Lp, Ap = np.array(Lp), np.array(Ap)
print(f"  ordinary run: eta = {etap:.4f}, alpha_0 = {a0p:.3f}, loss {Lp[0]:.3f} -> {Lp[-1]:.2e}, min alpha_t/alpha_0 = {Ap.min() / a0p:.3f}")
check("ordinary run (no guarantee): loss -> 0 and the last-layer features stay full rank", Lp[-1] < 1e-8 and Ap.min() > 0)
# notes' worked example
F = np.diag([1., 2.]); w = np.zeros(2); yv = np.ones(2); w2 = w - .1 * F.T @ (F @ w - yv)
check("notes' example: one step gives w = (0.1, 0.2), loss 0.585", np.allclose(w2, [.1, .2]) and abs(.5 * np.sum((F @ w2 - yv) ** 2) - .585) < 1e-12)
save_json("nguyen2021proof", "data.json", {"beta": round(beta, 4), "a0": round(float(a0), 5), "eta": float(f"{eta:.4g}"), "L": sig(L[::10], 4), "A": sig(A[::10], 4),
                                          "B": sig(bound[::10], 4), "rate": float(f"{eta * a0 ** 2 / 8:.3g}"),
                                          "etap": round(float(etap), 5), "a0p": round(float(a0p), 5), "Lp": sig(np.maximum(Lp[::10], 1e-30), 4), "Ap": sig(Ap[::10], 4), "Fp": Fp, "Y": Y.ravel().tolist()})
done()
