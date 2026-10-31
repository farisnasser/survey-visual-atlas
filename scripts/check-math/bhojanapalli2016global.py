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

# in-page notes (papers/bhojanapalli2016global/notes.md): Sec. 2 example, Sec. 8 counterexample and the primer's numbers
fb = lambda p: (p[0] ** 2 - 1) ** 2 + 2 * p[0] ** 2 * p[1] ** 2 + p[1] ** 4
gb = lambda p: np.array([4 * p[0] * (p[0] ** 2 + p[1] ** 2 - 1), 4 * p[1] * (p[0] ** 2 + p[1] ** 2)])
Xs = np.diag([1., 0])
check("notes Sec. 2: f(a,b) = ||UU^T - X*||_F^2 expands as (a^2-1)^2 + 2a^2b^2 + b^4",
      all(np.isclose(np.sum((np.outer(u, u) - Xs) ** 2), fb(u)) for u in rand_pts(20, -2, 2)))
check_grad("notes Sec. 2: gradient 4a(a^2+b^2-1), 4b(a^2+b^2)", fb, gb, rand_pts(20, -2, 2))
check_hess_at("notes Sec. 2: Hessian at the origin diag(-4, 0)", fb, [0, 0], [[-4, 0], [0, 0]])
check("notes Sec. 2: f(0,0) = 1, f(0.1,0) = 0.9801, f(+-1,0) = 0", fb([0, 0]) == 1 and np.isclose(fb([.1, 0]), .9801) and fb([1, 0]) == 0 and fb([-1, 0]) == 0)
f5 = lambda X: (X[0, 0] + X[1, 1] - 1) ** 2 + (X[0, 0] - 1) ** 2 + X[0, 1] ** 2 + X[1, 0] ** 2
check("notes Sec. 8 (paper Example 1): f(diag(1,0)) = 0, f(diag(0,1)) = 1", f5(np.diag([1., 0])) == 0 and f5(np.diag([0., 1])) == 1)
Rb = np.random.default_rng(58)
# the paper's claim that diag(0,1) is a rank-one local minimum is false: rank-one PSD matrices
# [[s, sqrt(s(1-s))], [sqrt(s(1-s)), 1-s]] tend to it and have loss 1 - s^2 (notes Sec. 8 correction)
cx = []
for s_ in (.1, .01, .001):
    b_ = np.sqrt(s_ * (1 - s_)); Xc = np.array([[s_, b_], [b_, 1 - s_]])
    cx.append(np.linalg.matrix_rank(Xc, 1e-12) == 1 and np.linalg.eigvalsh(Xc).min() > -1e-12 and np.isclose(f5(Xc), 1 - s_ ** 2) and f5(Xc) < 1)
check("notes Sec. 8 correction: rank-one PSD neighbours of diag(0,1) have loss 1 - s^2 < 1 (not a local min)", all(cx))
f6 = lambda X: (X[0, 0] + X[1, 1] + X[2, 2] - 1) ** 2 + (X[0, 0] - 1) ** 2 + (X[1, 1] - X[2, 2]) ** 2 + sum(X[i, j] ** 2 for i in range(3) for j in range(3) if i != j)
cx2 = []
for s_ in (.1, .01):
    b_ = np.sqrt(s_ * (.5 - s_ / 2)); Xc = np.array([[s_, b_, 0], [b_, .5 - s_ / 2, 0], [0, 0, .5 - s_ / 2]])
    cx2.append(np.linalg.matrix_rank(Xc, 1e-12) == 2 and np.linalg.eigvalsh(Xc).min() > -1e-12 and np.isclose(f6(Xc), 1 - s_))
check("paper Example 2 fails too: rank-2 PSD neighbours of diag(0,1/2,1/2) have loss 1 - s < 1 = f there", all(cx2) and np.isclose(f6(np.diag([0, .5, .5])), 1))
check("primer idea 1: n=100, r=5: nr = 500 vs n(n+1)/2 = 5050", 100 * 5 == 500 and 100 * 101 // 2 == 5050)
check("primer idea 1: U = (1,0), (-1,0) zero loss; midpoint U = 0 has loss 1", fb([1, 0]) == 0 and fb([-1, 0]) == 0 and fb([0, 0]) == 1)
Z = np.outer(Rb.normal(size=6), Rb.normal(size=6)); m_ = 200
ratios = np.array([np.mean(np.einsum('kij,ij->k', Rb.normal(size=(m_, 6, 6)), Z) ** 2) / np.sum(Z ** 2) for _ in range(2000)])
check("primer idea 2: (1/m) sum <A_i,Z>^2 / ||Z||_F^2 has mean 1 and sd sqrt(2/m) = 0.1 at m = 200",
      abs(ratios.mean() - 1) < .01 and abs(ratios.std() - .1) < .008 and np.isclose(np.sqrt(2 / 200), .1), f"mean {ratios.mean():.3f}, sd {ratios.std():.3f}")
done()
