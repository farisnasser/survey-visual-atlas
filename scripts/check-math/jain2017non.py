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
done()
