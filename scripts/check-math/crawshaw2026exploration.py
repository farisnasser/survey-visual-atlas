"""Crawshaw, Modi, Liu, Gower 2026: an exploration of non-Euclidean gradient descent (Muon and its variants).

LMO(v) = argmin_{|u| <= 1} <v, u>;  dual norm |v|_* = -<v, LMO(v)>.
Prop. 3.1 (CSD): w+ = w + eta LMO(m).  Prop. 3.2 (RSD): w+ = w + eta |m|_* LMO(m).
Spectral norm on a matrix: LMO(M) = -polar(M) = -U V^T, dual norm = nuclear norm. Muon = CSD with the spectral norm.
Diagonal 2x2 weight W = diag(a, b): spectral norm = max(|a|, |b|), polar(diag(g1, g2)) = diag(sign g1, sign g2),
nuclear norm = |g1| + |g2|: on the diagonal, Muon is sign descent and spectral RSD is |g|_1-scaled sign descent.
"""
from common import *

print("crawshaw2026exploration")
rng = np.random.default_rng(0)


def polar(M):
    U, s, Vt = np.linalg.svd(M)
    return U @ Vt


# notes example G = diag(3, 1)
G = np.diag([3., 1.])
check("notes: Frobenius unit step -G/|G|_F gives <G, D> = -sqrt(10)", abs(np.sum(G * (-G / np.linalg.norm(G))) + np.sqrt(10)) < 1e-12)
check("notes: spectral LMO -polar(G) = -I gives <G, D> = -4 = -|G|_nuc", np.allclose(polar(G), np.eye(2)) and abs(np.sum(G * -polar(G)) + 4) < 1e-12)
# LMO optimality for random 2x2 matrices: no matrix with spectral norm <= 1 does better than -polar(G)
worst = 0
for _ in range(200):
    M = rng.normal(size=(2, 2)); best = np.sum(M * -polar(M))
    for _ in range(200):
        U = rng.normal(size=(2, 2)); U /= np.linalg.norm(U, 2)
        worst = max(worst, best - np.sum(M * U))
check("spectral LMO: -polar(M) minimizes <M, U> over |U|_2 <= 1 (random search)", worst <= 1e-12, f"max improvement found {worst:.1e}")
M = rng.normal(size=(2, 2))
check("dual of the spectral norm is the nuclear norm: -<M, LMO(M)> = sum of singular values", abs(np.sum(M * polar(M)) - np.linalg.svd(M, compute_uv=False).sum()) < 1e-12)
d = np.array([.7, -.2])
check("diagonal case: polar(diag(g)) = diag(sign g), nuclear = |g|_1", np.allclose(polar(np.diag(d)), np.diag(np.sign(d))) and abs(np.linalg.svd(np.diag(d), compute_uv=False).sum() - np.abs(d).sum()) < 1e-12)

# 3D toy: f(a, b) = 1/2 [a b] H [a b]^T with an ill-conditioned, coupled H (W = diag(a, b))
H = np.array([[4., .6], [.6, .3]]); CEN = np.array([.6, -.5])
f = lambda w: .5 * (w - CEN) @ H @ (w - CEN)
g = lambda w: H @ (w - CEN)
check_grad("toy quadratic", f, g, rand_pts(4, -2, 2))
W0 = np.array([-1.6, 1.5])


def run(kind, eta, T=400):
    w = W0.copy(); out = []
    for _ in range(T):
        gr = g(w)
        if kind == "gd": w = w - eta * gr                                   # RSD, l2 (Euclidean)
        elif kind == "muon": w = w - eta * np.sign(gr)                      # CSD, spectral = l_inf on the diagonal
        elif kind == "rsd": w = w - eta * np.abs(gr).sum() * np.sign(gr)    # RSD, spectral: dual (nuclear) norm times LMO
        out.append(f(w))
    return np.array(out), w


Lg, _ = run("gd", .4); Lm, wm = run("muon", .05); Lr, wr = run("rsd", .1)
print(f"    final loss: GD {Lg[-1]:.2e}, Muon-style CSD {Lm[-1]:.2e}, spectral RSD {Lr[-1]:.2e}")
check("CSD (fixed-length steps) stalls at a level set by eta: never below ~eta^2 scale", Lm[-50:].min() > 1e-6 and Lm[-1] < .05, f"late min {Lm[-50:].min():.1e}")
check("RSD (steps scaled by the dual norm) keeps converging (loss 1e-9 after 400 steps)", Lr[-1] < 1e-8 and Lr[-1] < Lr[200] * 1e-2, f"{Lr[-1]:.1e}")
check("Muon/CSD moves both coordinates by exactly eta per step", np.allclose(np.abs(np.diff(np.array([W0, W0 - .05 * np.sign(g(W0))]), axis=0)), .05))
# in-page notes (papers/crawshaw2026exploration/notes.md): primer numbers
check("primer 1: I_2 has spectral norm 1, Frobenius norm sqrt 2", abs(np.linalg.norm(np.eye(2), 2) - 1) < 1e-12 and abs(np.linalg.norm(np.eye(2)) - np.sqrt(2)) < 1e-12)
gv = np.array([3., -1.])
check("primer 2: g = (3, -1): Euclidean decrease sqrt 10, max-norm step -sign(g) = (-1, 1) gives 4 = |g|_1", abs(gv @ (gv / np.linalg.norm(gv)) - np.sqrt(10)) < 1e-12 and np.allclose(-np.sign(gv), [-1, 1]) and gv @ np.sign(gv) == 4)
cand = np.random.default_rng(1).uniform(-1, 1, size=(20000, 2))
check("primer 2: no step in the max-norm ball beats -sign(g)", (cand @ gv).min() >= -4 - 1e-12)
Gp = np.array([[1., 1.], [0., 1.]])
U_, s_, Vt_ = np.linalg.svd(Gp); Pp = U_ @ Vt_
check("primer 3: singular values 1.618, 0.618; polar = [[0.894, 0.447], [-0.447, 0.894]]; <G, polar G> = 2.236", np.allclose(s_, [1.618, .618], atol=1e-3) and np.allclose(Pp, [[.894, .447], [-.447, .894]], atol=1e-3) and abs(np.sum(Gp * Pp) - np.sqrt(5)) < 1e-12)
X = Gp / np.linalg.norm(Gp)
for _ in range(60):
    X = 1.5 * X - .5 * X @ X.T @ X
check("primer 3: cubic Newton-Schulz iteration converges to the polar factor", np.allclose(X, Pp, atol=1e-10))
gq = np.array([3., 4.])
check("primer 4: eta = 0.1, g = (3, 4): CSD moves 0.1, RSD moves 0.5", abs(np.linalg.norm(.1 * gq / np.linalg.norm(gq)) - .1) < 1e-12 and abs(np.linalg.norm(.1 * gq) - .5) < 1e-12)
done()
