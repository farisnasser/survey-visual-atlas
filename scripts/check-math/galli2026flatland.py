"""Galli et al. 2026: monotone vs nonmonotone equality line search; eta*lambda_max vs 2; the 2/K floor."""
from common import *

print("galli2026flatland")
# test function: a curved narrow valley (non-convex), bounded below
A = 6.0


def f(p):
    x, y = p
    return .25 * (x * x - 1) ** 2 + .5 * A * (y - .3 * x * x) ** 2


def g(p):
    x, y = p
    r = y - .3 * x * x
    return np.array([x * (x * x - 1) - A * r * .6 * x, A * r])


def Hs(p):
    x, y = p
    r = y - .3 * x * x
    return np.array([[3 * x * x - 1 + A * (.36 * x * x) - A * r * .6, -A * .6 * x], [-A * .6 * x, A]])


check_grad("valley", f, g, rand_pts(40, -1.8, 1.8))
for p in rand_pts(5, -1.5, 1.5, seed=2):
    assert np.allclose(Hs(p), fd_hess(f, p), atol=1e-3)
check("analytic Hessian matches", True)


def eq_ls(w, eta0, R, c=1e-4, beta=.5):
    """Algorithm 2 (equality line search) with reference R: extrapolate by 1/beta while accepted, else backtrack."""
    gr = g(w)
    acc = lambda e: f(w - e * gr) <= R - c * e * gr @ gr
    eta = eta0
    if acc(eta):
        while acc(eta / beta) and eta < 1e3:
            eta /= beta
    else:
        while not acc(eta):
            eta *= beta
    return eta


def run(mode, w0, steps=400, xi=.85):
    w = np.array(w0, float)
    eta, Q, R = 1.0, 1.0, f(w)
    out = []
    for k in range(steps):
        Rk = f(w) if mode == "LS" else R
        eta = eq_ls(w, eta, Rk)
        lam = eig(Hs(w))[-1]
        out.append((f(w), eta, eta * lam, np.linalg.norm(g(w))))
        w = w - eta * g(w)
        if mode == "NLS":
            Qn = xi * Q + 1
            R = (xi * Q * R + f(w)) / Qn
            Q = Qn
    return np.array(out), w


w0 = [-1.6, 1.4]
for mode in ["LS", "NLS"]:
    o, w = run(mode, w0)
    inc = int(np.sum(np.diff(o[:, 0]) > 1e-12))
    print(f"  {mode}: final f={o[-1, 0]:.2e} |g|={o[-1, 3]:.1e}; loss increases {inc}; median eta*lam_max {np.median(o[:, 2]):.2f}; max {o[:, 2].max():.2f}")
oL, _ = run("LS", w0)
oN, wN = run("NLS", w0)
check("monotone LS never increases f", np.all(np.diff(oL[:, 0]) <= 1e-12))
check("NLS accepts temporary increases (nonmonotone)", np.sum(np.diff(oN[:, 0]) > 1e-12) > 3)
check("NLS operates at/above eta*lambda_max ~ 2 often", np.mean(oN[:, 2] > 1.9) > .3, f"fraction {np.mean(oN[:, 2] > 1.9):.2f}")
check("NLS still drives the gradient to 0 (Thm 2.9)", oN[-1, 3] < 1e-4)
# 2/K floor: MSE over K outputs with a final bias -> lambda_max(H) >= 2/K
rng = np.random.default_rng(0)


def mlp_loss(theta, X, Y, K, h):
    d = X.shape[1]
    i = 0
    W1 = theta[i:i + h * d].reshape(h, d); i += h * d
    b1 = theta[i:i + h]; i += h
    W2 = theta[i:i + K * h].reshape(K, h); i += K * h
    b2 = theta[i:i + K]
    Z = np.tanh(X @ W1.T + b1) @ W2.T + b2
    return np.mean(np.mean((Z - Y) ** 2, 1))


for K in [2, 4, 8]:
    X = rng.normal(size=(30, 2))
    Y = np.eye(K)[rng.integers(0, K, 30)]
    h = 4
    npar = h * 2 + h + K * h + K
    worst = 1e9
    for trial in range(6):
        th = rng.normal(0, 1.5, npar)
        lam = eig(fd_hess(lambda t: mlp_loss(t, X, Y, K, h), th, h=1e-3))[-1]
        worst = min(worst, lam)
    print(f"  K={K}: smallest lambda_max over random weights {worst:.4f} >= 2/K = {2 / K:.4f}")
    check(f"K={K}: lambda_max >= 2/K", worst >= 2 / K - 1e-3)
    # "Flatland" point: hidden weights 0, output bias = class means -> stationary, top-K eigenvalues = 2/K, negative ones too
    th = np.zeros(npar)
    th[-K:] = Y.mean(0)
    fl = lambda t: mlp_loss(t, X, Y, K, h)
    ev = eig(fd_hess(fl, th, h=1e-3))
    check(f"K={K}: flat point is stationary with top-K eigenvalues = 2/K and a negative eigenvalue",
          np.max(np.abs(fd_grad(fl, th))) < 1e-8 and np.allclose(ev[-K:], 2 / K, atol=1e-4) and ev[0] < -1e-3,
          f"lam_min {ev[0]:.3f}")
# in-page notes (papers/galli2026flatland/notes.md): Sec. 1 multipliers, Sec. 2 accepted loss increase, Sec. 3 2/K bound
check("notes: lambda = 4: eta = 0.40 -> -0.6, 0.50 -> -1, 0.55 -> -1.2", np.allclose([1 - 4 * .4, 1 - 4 * .5, 1 - 4 * .55], [-.6, -1, -1.2]))
xp = 1 - 2.1
check("notes: x+ = -1.1, f = 0.605 > 0.5; R - c eta |g|^2 = 0.70 - 0.021 = 0.679 >= 0.605", np.isclose(xp, -1.1) and np.isclose(.5 * xp ** 2, .605) and np.isclose(.7 - .01 * 2.1, .679) and .5 * xp ** 2 <= .679)
for K in [2, 10, 26, 100]:
    Rb = np.random.default_rng(K); z0, yk = Rb.normal(size=K), Rb.normal(size=K)
    fb = lambda b: np.mean((z0 + b - yk) ** 2)
    assert np.allclose(fd_hess(fb, np.zeros(K)), 2 / K * np.eye(K), atol=1e-5)
check("notes: Hessian of K^-1 sum (z_j + b_j - y_j)^2 in b is (2/K) I; 2/10 = 0.2, 2/100 = 0.02", np.isclose(2 / 10, .2) and np.isclose(2 / 100, .02))
Hf = np.array([[.2, .3, .1], [.3, -.5, .2], [.1, .2, .7]])
check("notes: lambda_max(H) >= lambda_max of a diagonal block (Rayleigh quotient)", eig(Hf)[-1] >= eig(Hf[:2, :2])[-1] and eig(Hf)[-1] >= Hf[2, 2])
# primer: Armijo boundary 2(1 - c)/lambda; halving/doubling lands at >= (1 - c)/lambda; Zhang-Hager average; saddle with small sharpness
c = .01
etas = np.linspace(1e-4, 2.5, 250001)
ok = .5 * (1 - etas) ** 2 <= .5 - c * etas + 1e-15
check("primer: f = x^2/2 at x = 1 passes Armijo exactly for eta <= 2(1 - c) = 1.98", np.isclose(etas[ok].max(), 1.98, atol=2e-5) and ok[etas <= 1.98].all())
def equality_ls(lam, eta0, c=.01, x=1.):
    acc = lambda e: .5 * lam * (x - e * lam * x) ** 2 <= .5 * lam * x * x - c * e * (lam * x) ** 2
    e = eta0
    if acc(e):
        while acc(2 * e): e *= 2
    else:
        while not acc(e): e /= 2
    return e
res = [(lam, e0, equality_ls(lam, e0)) for lam in [.3, 1., 4., 37.] for e0 in [1e-3, .07, 1., 50.]]
check("primer: halving/doubling equality search on a quadratic gives (1 - c)/lambda <= eta <= 2(1 - c)/lambda",
      all((1 - c) / lam - 1e-12 <= e <= 2 * (1 - c) / lam + 1e-12 for lam, _, e in res))
Q0, R0 = 1., 1.0; Q1 = .85 * Q0 + 1; R1 = (.85 * Q0 * R0 + .5) / Q1
check("primer: Zhang-Hager reference with xi = 0.85 after losses 1.0, 0.5 is 1.35/1.85 ~ 0.73", np.isclose(Q1, 1.85) and np.isclose(R1, .73, atol=.005), f"{R1:.4f}")
Hs = np.array([[.2, .3], [.3, -.5]])
check("primer: H = [[0.2, 0.3], [0.3, -0.5]] has eigenvalues ~0.31, ~-0.61 (sharpness >= 0.2, a saddle)", np.allclose(eig(Hs), [-.61, .31], atol=.005) and eig(Hs)[-1] >= .2)
done()
