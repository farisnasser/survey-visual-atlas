"""Staib, Reddi, Kale, Kumar, Sra 2019: adaptive methods rescale gradient noise to be isotropic near stationary points.

Replicates the Sec. 5 experiment: f_i(x) = 1/2 x^T H x + b_i^T x + |x|_10^10, H = diag(1, -0.1), E b = 0, Cov b = diag(1, 0.01)
(here b_i = (+-1, +-0.1), all four sign pairs, which gives exactly that covariance). SGD and diagonal RMSProp with
beta = 1 - eta^(2/3), both started at the saddle 0, 10^4 iterations.
Eq. (3): with A = G^(-1/2), G = E[g g^T], the noise xi = A (g - grad f) has Cov(xi) = I - G^(-1/2) grad grad^T G^(-1/2).
"""
from common import *

print("staib2019escaping")
Hd = np.array([1., -.1])
Bv = np.array([[1, .1], [1, -.1], [-1, .1], [-1, -.1]])
f = lambda p: .5 * (Hd * p * p).sum() + (p ** 10).sum()
g = lambda p: Hd * p + 10 * p ** 9
check_grad("f", f, g, rand_pts(20, -.8, .8))
check("Cov(b) = diag(1, 0.01), E b = 0", np.allclose(Bv.mean(0), 0) and np.allclose(Bv.T @ Bv / 4, np.diag([1, .01])))
x2 = .01 ** (1 / 8)
check("minima at (0, +-0.01^(1/8)) = (0, +-0.5623)", np.allclose(g(np.array([0, x2])), 0), f"f* = {f(np.array([0, x2])):.5f}")
# Eq. (3) at a point with small gradient
p = np.array([.05, .03]); gr = g(p); S = gr + Bv
G = S.T @ S / 4
w, V = np.linalg.eigh(G); Gm = V @ np.diag(w ** -.5) @ V.T
xi = (S - gr) @ Gm.T
check("Eq. (3): Cov(A(g - grad f)) = I - G^-1/2 grad grad^T G^-1/2", np.allclose(xi.T @ xi / 4, np.eye(2) - Gm @ np.outer(gr, gr) @ Gm, atol=1e-12))
check("notes example: Sigma = diag(9, 1) -> A = diag(1/3, 1) whitens the noise", np.allclose(np.diag([9, 1]) ** .5 @ np.diag([1 / 3, 1]), np.eye(2)))


def run(kind, eta, seed, T=10000, eps=1e-8):
    R = np.random.default_rng(seed); x = np.zeros(2)
    beta = 1 - eta ** (2 / 3); v = np.zeros(2)
    W = int(round(eta ** (-2 / 3)))                     # burn-in: estimate the preconditioner at x0 first
    for _ in range(W):
        gg = g(x) + Bv[R.integers(4)]; v = beta * v + (1 - beta) * gg * gg
    F = np.empty(T); esc = None
    for t in range(T):
        gg = g(x) + Bv[R.integers(4)]
        if kind == "rms":
            v = beta * v + (1 - beta) * gg * gg; x = x - eta * gg / np.sqrt(v + eps)
        else:
            x = x - eta * gg
        F[t] = f(x)
        if esc is None and abs(x[1]) > .4:
            esc = t
    return F, esc if esc is not None else T


cases = {"sgd .001": ("sgd", .001), "rms .001": ("rms", .001), "sgd .01": ("sgd", .01)}
res = {k: [run(*v, s) for s in range(20)] for k, v in cases.items()}
esc = {k: float(np.median([e for _, e in v])) for k, v in res.items()}
tail = {k: float(np.median([np.mean(F[-2000:]) for F, _ in v])) for k, v in res.items()}
print("  median iterations until |x2| > 0.4:", {k: int(v) for k, v in esc.items()})
print("  median of mean f over the last 2000 iterations:", {k: f"{v:.5f}" for k, v in tail.items()})
check("RMSProp (eta = 0.001) escapes much faster than SGD with the same step", esc["rms .001"] * 3 < esc["sgd .001"])
check("SGD with eta = 0.01 escapes about as fast as RMSProp with 0.001 (within 3x)", esc["sgd .01"] < 3 * esc["rms .001"])
check("but SGD with eta = 0.01 ends higher (bounces in the noise) than RMSProp with 0.001", tail["sgd .01"] > tail["rms .001"], f"{tail['sgd .01']:.5f} vs {tail['rms .001']:.5f}")
Fm = {k: sig(np.median([F[::50] for F, _ in v], 0), 4) for k, v in res.items()}
save_json("staib2019escaping", "data.json", {"esc": esc, "tail": {k: round(v, 6) for k, v in tail.items()}, "curves": Fm, "fstar": round(float(f(np.array([0, x2]))), 6)})
done()
