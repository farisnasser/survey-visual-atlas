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
# in-page notes (papers/nguyen2021proof/notes.md): F = diag(1, 2), W = 0, Y = (1, 1), eta = 0.1
Fn, Yn, Wn = np.diag([1., 2.]), np.array([1., 1.]), np.zeros(2)
en = Fn @ Wn - Yn; gn = Fn.T @ en; Wp = Wn - .1 * gn; Phi = lambda W_: .5 * np.sum((Fn @ W_ - Yn) ** 2)
check("notes example: e = (-1,-1), grad = (-1,-2), W+ = (0.1, 0.2), outputs (0.1, 0.4)", np.allclose(en, [-1, -1]) and np.allclose(gn, [-1, -2]) and np.allclose(Wp, [.1, .2]) and np.allclose(Fn @ Wp, [.1, .4]))
check("notes example: loss 1 -> 0.585", np.isclose(Phi(Wn), 1) and np.isclose(Phi(Wp), .585))
Rq = np.random.default_rng(10); okg = True
for _ in range(300):
    Nn, mm = Rq.integers(2, 6), Rq.integers(6, 10)
    F_ = Rq.normal(size=(Nn, mm)); W_ = Rq.normal(size=(mm, 2)); Y_ = Rq.normal(size=(Nn, 2))
    G_ = F_.T @ (F_ @ W_ - Y_); smin = np.linalg.svd(F_, compute_uv=False)[-1]
    okg &= np.sum(G_ ** 2) >= 2 * smin ** 2 * .5 * np.sum((F_ @ W_ - Y_) ** 2) - 1e-9
check("notes: ||grad_WL Phi||_F^2 >= 2 sigma_min(F_{L-1})^2 Phi (N <= m)", okg)
# primer idea 1: 3 examples, 2 features -> some residual e != 0 has F^T e = 0
F32 = Rq.normal(size=(3, 2)); e0 = np.linalg.svd(F32.T)[2][-1]
check("primer: N = 3 > 2 features leaves a residual with zero last-layer gradient", np.linalg.norm(e0) > .99 and np.allclose(F32.T @ e0, 0))
# primer idea 2: 0.99^t < 0.01 first at t = 459
check("primer: 0.99^459 < 0.01 <= 0.99^458", .99 ** 459 < .01 <= .99 ** 458)
# primer idea 3: ReLU 1-Lipschitz; Weyl: sigma_min(diag(1,2) + E) >= 0.6 when ||E||_2 = 0.4
aa, bb = Rq.normal(size=1000), Rq.normal(size=1000)
check("primer: |relu(a) - relu(b)| <= |a - b|", np.all(np.abs(np.maximum(aa, 0) - np.maximum(bb, 0)) <= np.abs(aa - bb) + 1e-15))
smins = []
for _ in range(2000):
    E_ = Rq.normal(size=(2, 2)); E_ *= .4 / np.linalg.norm(E_, 2)
    smins.append(np.linalg.svd(np.diag([1., 2.]) + E_, compute_uv=False)[-1])
check("primer: sigma_min(diag(1,2) + E) >= 0.6 for ||E||_2 = 0.4 (Weyl)", min(smins) >= .6 - 1e-12, f"min {min(smins):.4f}")
# primer idea 4: relu(beta z) = beta relu(z); L = 3: doubling beta multiplies alpha0^2 by 16 and a degree-2 polynomial with nonnegative coefficients by at most 4
zz = Rq.normal(size=50)
check("primer: relu(beta z) = beta relu(z) for beta > 0", np.allclose(np.maximum(2.7 * zz, 0), 2.7 * np.maximum(zz, 0)))
X3 = Rq.normal(size=(4, 5)); W1, W2 = Rq.normal(size=(5, 6)), Rq.normal(size=(6, 7))
a0 = lambda b_: np.linalg.svd(np.maximum(np.maximum(X3 @ (b_ * W1), 0) @ (b_ * W2), 0), compute_uv=False)[-1]
rhs = lambda b_: (b_ * np.linalg.norm(W1, 2) + 1) * (b_ * np.linalg.norm(W2, 2) + 1)
check("primer: L = 3: alpha0^2 scales by 16 when beta doubles; the right side of (14) by at most 4", np.isclose(a0(2.) ** 2 / a0(1.) ** 2, 16) and rhs(2.) / rhs(1.) <= 4)
done()
