"""Bhojanapalli, Neyshabur, Srebro 2016: PSD matrix sensing, n = 2, r = 1, so U = u in R^2 can be drawn.

y_i = <A_i, X*>,  X* = u* u*^T,  A_i = (G + G^T)/2 with G iid N(0,1) (so E<A, X>^2 = |X|_F^2 for symmetric X),
f(u) = (1/m) sum_i (u^T A_i u - y_i)^2,  grad f = (4/m) sum_i r_i A_i u.
RIP over all symmetric 2x2 matrices (rank <= 2 = 4r covers all of them here): delta = max |eig(Phi^T Phi / m) - 1| in an orthonormal basis.
"""
from common import *

print("bhojanapalli2016global")
us = np.array([1.0, .5])
Xs = np.outer(us, us)
E = [np.diag([1., 0]), np.diag([0., 1]), np.array([[0, 1], [1, 0]]) / np.sqrt(2)]


def draw(m, seed):
    G = np.random.default_rng(seed).normal(size=(m, 2, 2))
    return (G + G.transpose(0, 2, 1)) / 2


def mk(A):
    y = np.einsum("mij,ij->m", A, Xs)
    f = lambda u: np.mean((np.einsum("i,mij,j->m", u, A, u) - y) ** 2)
    g = lambda u: 4 / len(A) * np.einsum("m,mij,j->i", np.einsum("i,mij,j->m", u, A, u) - y, A, u)
    return f, g


def rip(A):
    Phi = np.array([[np.sum(Ai * Ej) for Ej in E] for Ai in A])
    ev = np.linalg.eigvalsh(Phi.T @ Phi / len(A))
    return float(max(abs(ev - 1)))


A = draw(8, 0); f, g = mk(A)
check_grad("sensing loss m = 8", f, g, rand_pts(20, -1.5, 1.5))
check("E<A, X>^2 = |X|_F^2 for this ensemble", abs(np.mean(np.einsum("mij,ij->m", draw(200000, 9), Xs) ** 2) - np.sum(Xs ** 2)) < .02 * np.sum(Xs ** 2))
# population (fully observed) analogue: f = |uu^T - X*|_F^2 has Hessian -4 X* at 0  (notes' example form)
fp = lambda u: np.sum((np.outer(u, u) - Xs) ** 2)
check_hess_at("population loss at u = 0 is -4 X*", fp, [0, 0], -4 * Xs)


def endpoints(A, n0=20, seed=1):
    """Vectorized GD from n0 random starts; returns per run (loss, recovery error, |grad|, lambda_min of the Hessian)."""
    f, g = mk(A); R = np.random.default_rng(seed)
    y = np.einsum("mij,ij->m", A, Xs)
    U = R.normal(size=(n0, 2)) * 1.2
    for _ in range(5000):
        r_ = np.einsum("ni,mij,nj->nm", U, A, U) - y
        U = U - .02 * 4 / len(A) * np.einsum("nm,mij,nj->ni", r_, A, U)
    out = []
    for u in U:
        out.append((f(u), np.linalg.norm(np.outer(u, u) - Xs), np.linalg.norm(g(u)), eig(fd_hess(f, u))[0]))
    return np.array(out)


rows = []
for m in [1, 2, 3, 4, 6, 10, 20, 50]:
    spur, wrong, draws_sp, dl = 0, 0, 0, []
    for seed in range(40):
        A = draw(m, 100 * m + seed); P = endpoints(A, 20, seed)
        st = (P[:, 2] < 1e-4) & (P[:, 3] > -1e-6)
        sp = int(np.sum(st & (P[:, 0] > 1e-6)))           # local min with positive loss
        spur += sp; draws_sp += sp > 0
        wrong += int(np.sum(st & (P[:, 0] <= 1e-6) & (P[:, 1] > 1e-3)))  # zero loss but wrong matrix
        dl.append(rip(A))
    rows.append([m, spur, wrong, float(np.median(dl)), draws_sp])
    print(f"  m = {m}: spurious local minima in {draws_sp}/40 draws ({spur}/800 runs), zero-loss-but-wrong {wrong}/800, median RIP delta {np.median(dl):.3f}")
mm = {r_[0]: r_ for r_ in rows}
check("m = 1 or 2: zero training loss does not identify X* (wrong zero-loss solutions)", mm[1][2] + mm[2][2] > 0)
check("spurious local minima exist for small m (RIP fails)", mm[3][1] + mm[4][1] + mm[6][1] > 0)
check("they become rarer as m grows: none found at m = 50", mm[50][1] == 0 and mm[50][2] == 0)
dm = [[m, float(np.median([rip(draw(m, 7000 + s)) for s in range(200)]))] for m in [3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]]
for m, d_ in dm:
    print(f"  m = {m}: median RIP constant {d_:.3f}")
check("RIP constant decreases with m and drops below 1/10 only for large m", dm[0][1] > dm[-1][1] and dm[-1][1] < .1)
save_json("bhojanapalli2016global", "data.json", {"rows": rows, "rip": dm, "us": us.tolist()})
done()
