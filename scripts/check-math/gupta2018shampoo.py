"""Gupta, Koren, Singer 2018: Shampoo, preconditioned stochastic tensor optimization.

Algorithm 1 (matrix case): L_t = L_{t-1} + G G^T, R_t = R_{t-1} + G^T G (L_0 = eps I, R_0 = eps I),
W_{t+1} = W_t - eta L_t^{-1/4} G_t R_t^{-1/4}.  Memory m^2 + n^2 instead of (mn)^2.
Lemma 4: (L kron R^T) vec(G) = vec(L G R) (row-major vec, vec(u v^T) = u kron v).
Lemma 8: eps I + (1/r) sum g g^T  <=  (eps I + sum G G^T)^{1/2} kron (eps I + sum G^T G)^{1/2}   (g = vec G, rank G <= r).
Theorem 7: regret <= sqrt(2r) D Tr(L_T^{1/4}) Tr(R_T^{1/4}) = O(sqrt T).
"""
from common import *

print("gupta2018shampoo")
rng = np.random.default_rng(0)


def mpow(M, p):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V * np.maximum(w, 1e-300) ** p) @ V.T


vec = lambda M: M.reshape(-1)          # row-major
G = rng.normal(size=(3, 4)); Lm = rng.normal(size=(3, 3)); Rm = rng.normal(size=(4, 4))
check("Lemma 4: (L kron R^T) vec(G) = vec(L G R)", np.allclose(np.kron(Lm, Rm.T) @ vec(G), vec(Lm @ G @ Rm)))
worst = np.inf
for trial in range(30):
    m, n, r, T, eps = 3, 4, int(rng.integers(1, 4)), 12, 1e-3
    Gs = [rng.normal(size=(m, r)) @ rng.normal(size=(r, n)) for _ in range(T)]
    left = eps * np.eye(m * n) + sum(np.outer(vec(g), vec(g)) for g in Gs) / r
    right = np.kron(mpow(eps * np.eye(m) + sum(g @ g.T for g in Gs), .5), mpow(eps * np.eye(n) + sum(g.T @ g for g in Gs), .5))
    worst = min(worst, eig(right - left)[0])
check("Lemma 8: Kronecker preconditioner dominates the (1/r) full-matrix one (30 random tests)", worst > -1e-8, f"min eigenvalue of the difference {worst:.2e}")
# notes example: G = diag(2, 1), one step, no damping -> L^-1/4 G R^-1/4 = I
G2 = np.diag([2., 1.]); L2 = G2 @ G2.T; R2 = G2.T @ G2
check("notes: L^{-1/4} G R^{-1/4} = I for G = diag(2, 1)", np.allclose(mpow(L2, -.25) @ G2 @ mpow(R2, -.25), np.eye(2)))
check("notes: 1000 x 1000 weights -> full preconditioner 1e12 entries vs Shampoo 2e6", 1000 ** 4 == 1e12 and 2 * 1000 ** 2 == 2e6)

# ---------- 3D tab: W in R^{1x2} on a rotated ill-conditioned quadratic ----------
EIG = np.array([8., .2]); WSTAR = np.array([.8, -.6]); D0 = np.array([-1.2, 1.1])   # start offset in the valley's own axes


def Qrot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def Arot(th):
    Q = Qrot(th)
    return Q @ np.diag(EIG) @ Q.T


def run2(method, th, eta, T=300, eps=1e-4):
    A = Arot(th); w = WSTAR + Qrot(th) @ D0; Ls, R = eps, eps * np.eye(2); H = eps * np.eye(2); d = eps * np.ones(2); out = []
    for t in range(T):
        g = A @ (w - WSTAR)
        if method == "diag":
            d = d + g * g; w = w - eta * g / np.sqrt(d)
        elif method == "shampoo":
            Ls = Ls + g @ g; R = R + np.outer(g, g); w = w - eta * Ls ** -.25 * (g @ mpow(R, -.25))
        elif method == "full":
            H = H + np.outer(g, g); w = w - eta * mpow(H, -.5) @ g
        out.append(.5 * (w - WSTAR) @ A @ (w - WSTAR))
    return np.array(out), w


ETA3 = {"diag": .5, "shampoo": .5, "full": .5}
res = {}
for th in [0, np.pi / 4]:
    for m in ETA3:
        L_, w = run2(m, th, ETA3[m]); res[(m, round(th, 3))] = L_[-1]
        print(f"    rotation {th:.3f} {m}: final loss {L_[-1]:.2e}")
check("Shampoo (1x2) and full AdaGrad are rotation-equivariant: same loss curve at 0 and 45 degrees",
      abs(np.log10(res[("shampoo", 0)]) - np.log10(res[("shampoo", .785)])) < .01 and max(res[("full", 0)], res[("full", .785)]) < 1e-25)
check("diagonal AdaGrad is not: it does much worse when the valley is rotated 45 degrees", res[("diag", .785)] > 100 * res[("diag", 0)], f"{res[('diag', 0)]:.1e} -> {res[('diag', .785)]:.1e}")

# ---------- matrix least squares: f(W) = 1/2 |A W B - C|^2, Kronecker-structured curvature ----------
m = n = 8
Ua, _ = np.linalg.qr(rng.normal(size=(m, m))); Ub, _ = np.linalg.qr(rng.normal(size=(n, n)))
Am = Ua @ np.diag(np.logspace(0, -1.2, m)) @ Ua.T; Bm = Ub @ np.diag(np.logspace(0, -1.2, n)) @ Ub.T
Wst = rng.normal(size=(m, n)); Cm = Am @ Wst @ Bm
f = lambda W: .5 * np.sum((Am @ W @ Bm - Cm) ** 2)
grad = lambda W: Am.T @ (Am @ W @ Bm - Cm) @ Bm.T
W0 = np.zeros((m, n))
check_grad("matrix least squares", lambda v: f(v.reshape(m, n)), lambda v: grad(v.reshape(m, n)).reshape(-1), [rng.normal(size=m * n) for _ in range(3)])


def runM(method, eta, T=1500, eps=1e-6):
    W = W0.copy(); L_, R_ = eps * np.eye(m), eps * np.eye(n); D = eps * np.ones((m, n)); H = eps * np.eye(m * n); out = []
    for t in range(T):
        Gm = grad(W)
        if method == "sgd": W = W - eta * Gm
        elif method == "diag": D = D + Gm * Gm; W = W - eta * Gm / np.sqrt(D)
        elif method == "shampoo": L_ = L_ + Gm @ Gm.T; R_ = R_ + Gm.T @ Gm; W = W - eta * mpow(L_, -.25) @ Gm @ mpow(R_, -.25)
        elif method == "full": H = H + np.outer(vec(Gm), vec(Gm)); W = W - eta * (mpow(H, -.5) @ vec(Gm)).reshape(m, n)
        out.append(f(W))
    return np.array(out)


best = {}
grids = {"sgd": [.5, 1, 1.5, 1.9], "diag": [.1, .3, 1, 2, 4], "shampoo": [.1, .3, 1, 2, 4], "full": [.1, .3, 1, 2, 4]}
for meth, gr in grids.items():
    curves = {e: runM(meth, e) for e in gr}
    ok = {e: c for e, c in curves.items() if np.all(np.isfinite(c))}
    e = min(ok, key=lambda e: ok[e][-1]); best[meth] = (e, ok[e])
    print(f"    {meth}: best eta {e}, final loss {ok[e][-1]:.3e}")
check("Shampoo beats diagonal AdaGrad and GD on Kronecker-structured curvature", best["shampoo"][1][-1] < best["diag"][1][-1] and best["shampoo"][1][-1] < best["sgd"][1][-1])
# in-page notes (papers/gupta2018shampoo/notes.md): memory numbers and the primer
check("notes: 1000 x 1000 weight: flattened preconditioner 1e12 entries, two factors 2e6", (1000 * 1000) ** 2 == 10 ** 12 and 2 * 1000 ** 2 == 2 * 10 ** 6)
check("primer 1: regret 100 after T = 10 000 rounds is 0.01 per round (sqrt T = 100)", np.sqrt(10000) == 100 and 100 / 10000 == .01)
A_ = np.array([[2., 1.], [1., 2.]])
Ah = mpow(A_, -.5)
check("primer 3: A = [[2,1],[1,2]] has eigenvalues 3, 1; A^(-1/2) = [[0.789, -0.211], [-0.211, 0.789]]", np.allclose(np.linalg.eigvalsh(A_), [1, 3]) and np.allclose(Ah, [[.789, -.211], [-.211, .789]], atol=5e-4) and np.allclose(Ah @ Ah @ A_, np.eye(2)))
check("primer 3: entrywise power gives [[0.707, 1], [1, 0.707]] instead", np.allclose(A_ ** -.5, [[.7071, 1], [1, .7071]], atol=1e-4))
rng_ = np.random.default_rng(9)
P1 = rng_.normal(size=(3, 3)); P1 = P1 @ P1.T + np.eye(3); P2 = rng_.normal(size=(4, 4)); P2 = P2 @ P2.T + np.eye(4)
check("primer 4: (L kron R)^p = L^p kron R^p for PSD L, R (p = -1/4)", np.allclose(mpow(np.kron(P1, P2), -.25), np.kron(mpow(P1, -.25), mpow(P2, -.25))))
save_json("gupta2018shampoo", "data.json", {"m": m, "n": n, "curves": {k: {"eta": v[0], "loss": sig(v[1][::5], 4)} for k, v in best.items()},
                                            "mem": {"sgd": m * n, "diag": m * n, "shampoo": m * m + n * n, "full": (m * n) ** 2}})
done()
