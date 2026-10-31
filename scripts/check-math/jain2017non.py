"""Jain & Kar 2017 (Foundations and Trends in ML 10(3-4); arXiv 1712.07897), Ch. 3: generalized projected gradient descent (gPGD,
Algorithm 2) x <- Proj_C(x - eta grad f(x)) over a non-convex set C. Theorem 3.3: if f is alpha-RSC and beta-RSS over C with
beta/alpha < 2 and eta = 1/beta, then f(x_T) <= f(x*) + eps after O(alpha/(2 alpha - beta) log(1/eps)) steps.
Toy: C = B0(1) in R^2 (the two coordinate axes, a non-convex set), f(w) = 1/2 (w - w*)^T A (w - w*), w* = (1.2, 0) in C.
For x, y in C the difference x - y can be any vector of R^2, so alpha, beta = extreme eigenvalues of A.
"""
from common import *

print("jain2017non")
WS = np.array([1.2, 0.])
proj = lambda z: np.array([z[0], 0.]) if abs(z[0]) >= abs(z[1]) else np.array([0., z[1]])     # keep the larger coordinate
res = {}
for name, c in [("good", .3), ("bad", .9)]:
    A = np.array([[1, c], [c, 1]]); al, be = eig(A)
    f = lambda w, A=A: .5 * (w - WS) @ A @ (w - WS); g = lambda w, A=A: A @ (w - WS)
    check_grad(f"f ({name})", f, g, rand_pts(10, -2, 2))
    w = np.array([0., 1.5]); hist = [f(w)]
    for t in range(60):
        w = proj(w - g(w) / be); hist.append(f(w))
    res[name] = (float(al), float(be), w.tolist(), hist)
    print(f"  {name}: alpha = {al:.2f}, beta = {be:.2f}, beta/alpha = {be / al:.2f}; gPGD (eta = 1/beta) from (0, 1.5) ends at {np.round(w, 4)} with f = {f(w):.4f}")
check("beta/alpha = 1.86 < 2: gPGD finds the optimum over C (f -> 0) linearly", res["good"][1] / res["good"][0] < 2 and res["good"][3][-1] < 1e-12)
check("beta/alpha = 19 > 2: gPGD stalls at a fixed point on the wrong axis, (0, 1.08), with f = 0.137", res["bad"][1] / res["bad"][0] > 2 and abs(res["bad"][2][1] - 1.08) < 1e-9 and abs(res["bad"][3][-1] - .1368) < 1e-3)
# linear rate in the good case: f_{t+1}/f_t
h = np.array(res["good"][3]); rat = h[6:12] / h[5:11]
print(f"  good case contraction f_(t+1)/f_t ~ {rat.mean():.3f}")
check("good case: geometric decrease", np.all(rat < .8))
save_json("jain2017non", "data.json", {k: {"alpha": v[0], "beta": v[1], "end": v[2], "f": [float(f"{q:.4g}") for q in v[3]]} for k, v in res.items()})

# in-page notes (papers/jain2017non/notes.md): sparse / low-rank projections (Sec. 2) and primer numbers
import itertools
def proj_sparse(z, s):
    out = np.zeros_like(z); idx = np.argsort(-np.abs(z))[:s]; out[idx] = z[idx]; return out
zp = np.array([.3, -2, 1.1, .05])
cands = [np.where(np.isin(np.arange(4), S), zp, 0) for S in itertools.combinations(range(4), 2)]
check("notes/primer: sparse projection keeps the s largest |entries|: (0.3,-2,1.1,0.05), s=2 -> (0,-2,1.1,0)",
      np.allclose(proj_sparse(zp, 2), [0, -2, 1.1, 0]) and min(np.linalg.norm(zp - c) for c in cands) == np.linalg.norm(zp - proj_sparse(zp, 2)))
Mr = np.random.default_rng(7).normal(size=(5, 4)); Ur, Sr, Vr = np.linalg.svd(Mr)
Pr = Sr[0] * np.outer(Ur[:, 0], Vr[0])
best_rand = min(np.linalg.norm(Mr - np.outer(a, b)) for a, b in zip(np.random.default_rng(8).normal(size=(3000, 5)), np.random.default_rng(9).normal(size=(3000, 4))))
check("notes: truncated SVD is the best rank-1 approximation (beats 3000 random rank-1 matrices)", np.linalg.norm(Mr - Pr) <= best_rand)
check("primer: best rank-1 approximation of diag(3,1) is diag(3,0), distance 1", np.allclose(Pr.shape, (5, 4)) and np.allclose(np.diag([3., 0]), 3 * np.outer([1, 0], [1, 0])) and abs(np.linalg.norm(np.diag([3., 1]) - np.diag([3., 0])) - 1) < 1e-15)
check("primer: midpoint of (1,0,0) and (0,1,0) has 2 nonzeros", np.count_nonzero((np.array([1, 0, 0]) + np.array([0, 1, 0])) / 2) == 2)
Aj = np.array([[1., 0, 1], [0, 1, 1]])
check("primer: A^T A has eigenvalues 0, 1, 3 and null direction (1,1,-1)", np.allclose(np.linalg.eigvalsh(Aj.T @ Aj), [0, 1, 3]) and np.allclose(Aj @ [1, 1, -1], 0))
ev2 = np.concatenate([np.linalg.eigvalsh(Aj[:, S].T @ Aj[:, S]) for S in itertools.combinations(range(3), 2)])
check("primer: over 2-sparse directions the Rayleigh quotient lies in [0.382, 2.618]", abs(ev2.min() - .382) < 5e-4 and abs(ev2.max() - 2.618) < 5e-4)
rngJ = np.random.default_rng(0); mJ, dJ, sJ = 50, 200, 3
AJ = rngJ.normal(0, 1 / np.sqrt(mJ), (mJ, dJ)); rr = []
for k in range(20000):
    wJ = np.zeros(dJ); idx = rngJ.choice(dJ, sJ, replace=False); wJ[idx] = rngJ.normal(size=sJ); rr.append(np.sum((AJ @ wJ) ** 2) / np.sum(wJ * wJ))
rr = np.array(rr); p1, p99 = np.percentile(rr, [1, 99])
check("primer: 50x200 Gaussian A: null space dim >= 150", dJ - np.linalg.matrix_rank(AJ) >= 150)
check("primer: |Aw|^2/|w|^2 over 20000 random 3-sparse w in [0.38, 1.97]; 98% in [0.60, 1.51]",
      abs(rr.min() - .38) < .005 and abs(rr.max() - 1.97) < .005 and abs(p1 - .60) < .005 and abs(p99 - 1.51) < .005, f"{rr.min():.3f} {rr.max():.3f} {p1:.3f} {p99:.3f}")
done()
