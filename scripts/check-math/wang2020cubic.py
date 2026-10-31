"""Wang, Zhou, Liang, Lan 2020: cubic regularization with momentum (CRm, Alg. 1).

s = argmin g^T s + 1/2 s^T H s + (M/6)|s|^3 (exact, secular equation),  y_{k+1} = x_k + s,
beta_{k+1} = min{beta, |grad f(y_{k+1})|, |y_{k+1} - x_k|},  v = y_{k+1} + beta_{k+1}(y_{k+1} - y_k),  x_{k+1} = argmin_{y, v} f.
"""
from common import *

print("wang2020cubic")


def cubic_step(g, H, M):
    lam, V = np.linalg.eigh(H); gb = V.T @ g
    lo = max(0, -2 * lam[0] / M) + 1e-14
    nr = lambda r: np.linalg.norm(gb / (lam + M * r / 2))
    if nr(lo) < lo:  # hard case (includes g = 0 at a saddle)
        Db = np.where(lam + M * lo / 2 > 1e-12, -gb / (lam + M * lo / 2), 0); Db[0] = np.sqrt(max(lo ** 2 - Db @ Db, 0))
        return V @ Db
    hi = lo + 1
    while nr(hi) > hi:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if nr(mid) > mid else (lo, mid)
    return V @ (-gb / (lam + M * hi / 2))


R = np.random.default_rng(0)
for t in range(8):
    g = R.normal(size=2) * (0 if t == 0 else 1); A = R.normal(size=(2, 2)); H = (A + A.T) / 2; M = R.uniform(.5, 3)
    m = lambda s: g @ s + .5 * s @ H @ s + M / 6 * np.linalg.norm(s) ** 3
    s = cubic_step(g, H, M)
    th = np.linspace(0, 2 * np.pi, 721); rr = np.linspace(0, 6, 1201)
    scan = min(m(r_ * np.array([np.cos(a), np.sin(a)])) for a in th[::4] for r_ in rr[::4])
    check(f"cubic subproblem solved globally (instance {t})", m(s) <= scan + 1e-6, f"m(s) = {m(s):.5f}, scan {scan:.5f}")

f = lambda p: .5 * (1 - p[0]) ** 2 + 5 * (p[1] - p[0] ** 2) ** 2
g = lambda p: np.array([-(1 - p[0]) - 20 * p[0] * (p[1] - p[0] ** 2), 10 * (p[1] - p[0] ** 2)])
H = lambda p: np.array([[1 - 20 * p[1] + 60 * p[0] ** 2, -20 * p[0]], [-20 * p[0], 10]])
check_grad("f", f, g, rand_pts(20, -2, 2))
for p in rand_pts(4, -2, 2, seed=3):
    check_hess_at(f"at {np.round(p, 2)}", f, p, H(p), tol=5e-3)
x0 = np.array([-.4, 1.4])
check("start has an indefinite Hessian", eig(H(x0))[0] < 0)


def run(M, beta, K=400, tol=1e-9):
    x = x0.copy(); yp = x.copy(); F = [f(x)]; B = []; used = []
    for k in range(K):
        y = x + cubic_step(g(x), H(x), M)
        if beta > 0:
            b = min(beta, np.linalg.norm(g(y)), np.linalg.norm(y - x)); v = y + b * (y - yp)
            xn = v if f(v) < f(y) else y; B.append(b); used.append(bool(f(v) < f(y)))
        else:
            xn = y
        yp = y; x = xn; F.append(f(x))
        if np.linalg.norm(g(x)) < tol:
            break
    return np.array(F), k + 1, B, used, x


rows = {}
# Hessian-Lipschitz constant of f on |x| <= 1.5: |d^3 f/dx^3| = 120|x| <= 180, so M = 180 satisfies M > L2 (Alg. 1 input)
for M in [60, 180]:
    Fc, kc, _, _, xc = run(M, 0)
    Fm, km, Bm, um, xm = run(M, .9)
    print(f"  M = {M}: CR {kc} iterations, CRm {km} (momentum candidate accepted {sum(um)} of {len(um)})")
    check(f"M = {M}: both reach the minimum (1, 1)", np.linalg.norm(xc - 1) < 1e-6 and np.linalg.norm(xm - 1) < 1e-6)
    if M >= 180:
        check(f"M = {M} > L2: CR and CRm are both monotone", np.all(np.diff(Fm) <= 1e-15) and np.all(np.diff(Fc) <= 1e-15))
    check(f"M = {M}: CRm needs no more iterations than CR on this run", km <= kc)
    rows[M] = {"cr": kc, "crm": km}
# saddle: CR moves from an exact saddle where the gradient is zero
fs = lambda p: .5 * p[0] ** 2 - .5 * p[1] ** 2 + .25 * p[1] ** 4
s = cubic_step(np.zeros(2), np.diag([1., -1.]), 2.0)
check("at a saddle with zero gradient the cubic step moves along the negative eigenvector (|s| = 2|lambda_min|/M = 1)", abs(abs(s[1]) - 1) < 1e-9 and abs(s[0]) < 1e-12 and fs(s) < 0)
# notes' 1D numbers
f1 = lambda x: -x * x / 2 + x ** 4 / 4
check("notes: f(0.8) = -0.2176, f(0.86) = -0.2330", abs(f1(.8) + .2176) < 1e-12 and abs(f1(.86) + .2330) < 1e-4)
save_json("wang2020cubic", "data.json", {"iters": rows})

# in-page notes (papers/wang2020cubic/notes.md): notes Secs. 1-2 examples, primer numbers
fw = lambda x_: -x_ ** 2 / 2 + x_ ** 4 / 4
m0 = lambda s_: -s_ ** 2 / 2 + 2 / 6 * abs(s_) ** 3
sgw = np.linspace(-3, 3, 600001)
check("notes Sec. 1: M = 2 cubic model -s^2/2 + |s|^3/3 has minima at +-1; f(1) = -1/4 < f(0) = 0; f'(0) = 0, f''(0) = -1", abs(abs(sgw[np.argmin(m0(sgw))]) - 1) < 1e-4 and np.isclose(fw(1.), -.25) and np.isclose((fw(1e-4) - 2 * fw(0) + fw(-1e-4)) / 1e-8, -1, atol=1e-3))
check("notes Sec. 2: v = 0.8 + 0.2*0.3 = 0.86; f(0.8) = -0.2176, f(0.86) ~ -0.2330; monotone step keeps 0.86", np.isclose(.8 + .2 * .3, .86) and np.isclose(fw(.8), -.2176) and abs(fw(.86) + .2330) < 5e-5 and fw(.86) < fw(.8))
check("primer: overshoot y = 1 from 0.5 with beta = 0.5 -> v = 1.25, f(1.25) ~ -0.171 > f(1) = -0.25", np.isclose(1 + .5 * .5, 1.25) and abs(fw(1.25) + .171) < 5e-4 and fw(1.25) > fw(1.))
check("primer: eps = 1e-3 -> eps^-2 = 1e6, eps^-1.5 ~ 3.2e4", np.isclose(1e-3 ** -2, 1e6) and abs(1e-3 ** -1.5 / 3.2e4 - 1) < .02)
ee = [.1]
for _ in range(3): ee.append(ee[-1] ** 2)
check("primer: quadratic convergence 0.1 -> 1e-2 -> 1e-4 -> 1e-8", np.allclose(ee, [.1, 1e-2, 1e-4, 1e-8]))
gq = lambda x_: x_ * (x_ ** 2 - 1)
check("primer: error bound for (x^2-1)^2/4 near 1: at 1.01 dist 0.01, grad ~0.0203; ratio -> 1/2", abs(gq(1.01) - .0203) < 5e-5 and abs(1e-6 / gq(1 + 1e-6) - .5) < 1e-5)
check("primer: x^4 at 0.1: dist 0.1, grad 0.004; ratio 1/(4x^2) unbounded", np.isclose(4 * .1 ** 3, .004) and .001 / (4 * .001 ** 3) > 1e5)
done()
