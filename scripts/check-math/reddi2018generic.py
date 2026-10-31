"""Reddi et al. 2018: generic framework (Alg. 1) = gradient-focused optimizer (SVRG) + Hessian-focused optimizer (HessianDescent).

HessianDescent (Alg. 3): unit v with v^T H v <= lambda_min + gamma/2, alpha = |v^T H v|/M, u = x - alpha sign(v^T grad f) v,
keep argmin{f(u), f(x)}.  Lemma 2: decrease >= h = gamma^3/(24 M^2) when lambda_min(H) <= -gamma.
"""
from common import *

print("reddi2018generic")
# Lemma 2 on f_g(x, y) = x^2/2 - (gamma/2) y^2 + (M/6)|y|^3  (Hessian M-Lipschitz), at the saddle 0
for gam, M in [(.5, 1), (.2, 2), (1, 3)]:
    fg = lambda p: .5 * p[0] ** 2 - .5 * gam * p[1] ** 2 + M / 6 * abs(p[1]) ** 3
    worst = np.inf
    for phi in np.linspace(0, np.pi / 2, 2001):
        v = np.array([np.sin(phi), np.cos(phi)])
        q = v @ np.diag([1, -gam]) @ v
        if q > -gam + gam / 2:
            continue
        u = -(abs(q) / M) * v
        worst = min(worst, fg([0, 0]) - min(fg(u), fg([0, 0])))
    exact = gam ** 3 / (3 * M ** 2)
    check(f"gamma={gam}, M={M}: exact eigenvector decrease = gamma^3/(3M^2)", abs(fg([0, 0]) - fg(-(gam / M) * np.array([0, 1.])) - exact) < 1e-12)
    check(f"gamma={gam}, M={M}: every admissible v decreases f by >= gamma^3/(24 M^2)", worst >= gam ** 3 / (24 * M ** 2), f"worst {worst:.4g} vs h {gam ** 3 / (24 * M ** 2):.4g}")

# framework run on f = x^2/2 - y^2/2 + y^4/4, finite sum f_i = f + a_i x + c_i (x^2 - y^2)/2 (all keep y = 0 invariant)
n = 50
R0 = np.random.default_rng(3)
a = R0.normal(0, 1, n); c = R0.normal(0, .4, n); a -= a.mean(); c -= c.mean()
f = lambda p: .5 * p[0] ** 2 - .5 * p[1] ** 2 + .25 * p[1] ** 4
g = lambda p: np.array([p[0], -p[1] + p[1] ** 3])
gi = lambda p, i: g(p) + np.array([a[i] + c[i] * p[0], -c[i] * p[1]])
check_grad("f", f, g, rand_pts(20, -1.5, 1.5))
Mlip = 6 * 1.5  # |f'''| <= 6|y| on |y| <= 1.5
L = 1 + 3 * 1.5 ** 2  # crude gradient-Lipschitz bound on the box
eta = 1 / (4 * L * n ** (2 / 3))
print(f"  SVRG step 1/(4 L n^(2/3)) = {eta:.4f}, M = {Mlip}")


def svrg_epoch(x, R, m=n):
    snap = x.copy(); mu = g(snap); its = []
    for _ in range(m):
        i = R.integers(0, n)
        x = x - eta * (gi(x, i) - gi(snap, i) + mu); its.append(x.copy())
    return its[-1], its[R.integers(0, m)]


def hess_descent(x, gam):
    Hm = np.array([[1, 0], [0, -1 + 3 * x[1] ** 2]])
    lam, V = np.linalg.eigh(Hm); v = V[:, 0]; q = v @ Hm @ v
    if lam[0] > -gam:
        return x, True  # certified: no gamma-negative curvature
    s = np.sign(v @ g(x)) or 1.0
    u = x - abs(q) / Mlip * s * v
    return (u if f(u) < f(x) else x), False


R = np.random.default_rng(0)
x = np.array([1.5, 0.0]); y = x.copy()
for t in range(400):
    y, _ = svrg_epoch(y, R)
x_svrg = y
check("SVRG alone from the saddle's stable set converges to the saddle", np.linalg.norm(x_svrg) < 1e-3 and abs(x_svrg[1]) < 1e-12)
x = np.array([1.5, 0.0]); hfo = 0; p = .5
for t in range(400):
    yy, zz = svrg_epoch(x, R)
    u = yy if R.uniform() < p else zz
    x, cert = hess_descent(u, .5)
    hfo += not cert
    if cert and np.linalg.norm(g(x)) < 1e-6:
        break
check("framework: HessianDescent steps leave the saddle; ends at a minimum", abs(f(x) + .25) < 1e-6, f"x = {np.round(x, 4)}, HFO moves {hfo}, rounds {t + 1}")

# in-page notes (papers/reddi2018generic/notes.md): notes Sec. 1 example, primer numbers
fr = lambda x_, y_: .5 * x_ ** 2 - .5 * y_ ** 2 + .25 * y_ ** 4
check("notes Sec. 1: grad f(0,0) = 0, Hessian diag(1,-1), v = (0,1) has v^T H v = -1", np.allclose(fd_grad(lambda p_: fr(*p_), [0., 0.]), 0, atol=1e-8) and np.allclose(fd_hess(lambda p_: fr(*p_), [0., 0.]), [[1, 0], [0, -1]], atol=1e-4))
check("notes Sec. 1: f(0, 0.1) = -0.005 + 0.000025 = -0.004975", np.isclose(fr(0, .1), -.004975) and np.isclose(-.005 + .000025, -.004975))
a_i = np.array([-1., 1.]); xs_, xt_ = .5, -.3
plain = xs_ - a_i; svrg = (xs_ - a_i) - (xt_ - a_i) + (xt_ - a_i.mean())
check("primer: f_i = (x - a_i)^2/2, a = +-1, x = 0.5: plain gradients -0.5/1.5 (variance 1), SVRG gives 0.5 every time", np.allclose(sorted(plain), [-.5, 1.5]) and np.isclose(plain.var(), 1) and np.allclose(svrg, .5))
for g_, M_ in [(1., 1.), (.3, 2.)]:
    fh = lambda y_: -g_ / 2 * y_ ** 2 + M_ / 6 * abs(y_) ** 3
    check(f"primer: step gamma/M along v gives -gamma^3/(3M^2) (gamma = {g_}, M = {M_}) >= Lemma 2's gamma^3/(24 M^2)", np.isclose(fh(g_ / M_), -g_ ** 3 / (3 * M_ ** 2)) and g_ ** 3 / (3 * M_ ** 2) >= g_ ** 3 / (24 * M_ ** 2))
check("primer: Delta = 10, theta = 0.01, T = 4000 -> Delta/(T theta) = 0.25", np.isclose(10 / (4000 * .01), .25))
done()
