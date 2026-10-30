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
done()
