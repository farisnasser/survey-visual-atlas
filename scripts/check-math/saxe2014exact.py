"""Saxe et al. 2014: E(a,b) = 1/2 (s - ab)^2, hyperbolas, sigmoid curves, orthogonal vs Gaussian products."""
from common import *

print("saxe2014exact")
s = 1.0
E = lambda p: .5 * (s - p[0] * p[1]) ** 2
Eg = lambda p: np.array([-(s - p[0] * p[1]) * p[1], -(s - p[0] * p[1]) * p[0]])
check_grad("E", E, Eg, rand_pts(40, -1.6, 1.6))
check_hess_at("origin", E, [0, 0], [[0, -1], [-1, 0]])
p = gd(Eg, [.3, .05], .005, 6000)
c = p[:, 0] ** 2 - p[:, 1] ** 2
check("small-step GD nearly conserves a^2-b^2", np.max(np.abs(c - c[0])) < 2e-3, f"drift {np.max(np.abs(c - c[0])):.1e}")
check("GD reaches ab = s", abs(p[-1, 0] * p[-1, 1] - s) < 1e-3)


def u_analytic(t, s, u0, tau=1.0):
    e = np.exp(2 * s * t / tau)
    return s * e / (e - 1 + s / u0)


for ss in [3, 2, 1, .5]:
    u0, dt, T = 1e-3, 1e-4, 8.0
    u = u0
    for _ in range(int(T / dt)):
        u += dt * 2 * u * (ss - u)
    check(f"Eq.12 matches the ODE for s={ss}", abs(u - u_analytic(T, ss, u0)) < 1e-3 * ss)
lr, ss, a0 = .01, 2.0, .03
p = gd(lambda q: np.array([-(ss - q[0] * q[1]) * q[1], -(ss - q[0] * q[1]) * q[0]]), [a0, a0], lr, 800)
uu = p[:, 0] * p[:, 1]
ua = u_analytic(np.arange(len(uu)) * lr, ss, a0 * a0)
check("discrete GD on a=b follows Eq.12 with t=k*lr", np.max(np.abs(uu - ua)) < .05, f"max dev {np.max(np.abs(uu - ua)):.3f}")
rng = np.random.default_rng(0)
N = 100


def prod_sv(D, kind):
    W = np.eye(N)
    for _ in range(D):
        M = rng.normal(0, 1 / np.sqrt(N), (N, N)) if kind == "g" else np.linalg.qr(rng.normal(size=(N, N)))[0]
        W = M @ W
    return np.linalg.svd(W, compute_uv=False)


for D in [1, 8, 32]:
    g, o = prod_sv(D, "g"), prod_sv(D, "o")
    print(f"  depth {D}: gaussian sv median {np.median(g):.3g} max {g.max():.3g} | orthogonal {o.min():.4f}..{o.max():.4f}")
check("orthogonal products keep all sv = 1", np.allclose(prod_sv(32, "o"), 1, atol=1e-8))
g = prod_sv(32, "g")
check("gaussian products: most sv tiny, a few large", np.median(g) < .05 and g.max() > 1)
# in-page notes (papers/saxe2014exact/notes.md): Sec. 1 gradients and the w = 0.01 example, Sec. 2 product rule,
# Sec. 4 dw/dt = w(1 - w^2), du/dt = 2u(1 - u) and its logistic solution, the factor-2 warning
Ls = lambda p: .5 * (1 - p[0] * p[1]) ** 2
gs = lambda p: np.array([-(1 - p[0] * p[1]) * p[1], -(1 - p[0] * p[1]) * p[0]])
check_grad("notes: dL/dw1 = -(1 - w1 w2) w2, dL/dw2 = -(1 - w1 w2) w1", Ls, gs, rand_pts(20, -2, 2))
p0 = np.array([.01, .01])
check("notes: w1 = w2 = 0.01 -> prediction 0.0001, loss ~0.5, |gradient| ~0.01", np.isclose(p0[0] * p0[1], 1e-4) and np.isclose(Ls(p0), .5, atol=1e-3) and np.allclose(np.abs(gs(p0)), .01, atol=1e-5), f"loss {Ls(p0):.5f}, grad {gs(p0)}")
wd = np.array([.7, 1.3, .4, 2.]); xd, yd = 1.5, .3
Ld = lambda w: .5 * (np.prod(w) * xd - yd) ** 2
check_grad("notes: dL/dw_i = (yhat - y) x prod_{j != i} w_j", Ld, lambda w: np.array([(np.prod(w) * xd - yd) * xd * np.prod(np.delete(w, i)) for i in range(len(w))]), [wd, wd[::-1]])
rate = lambda w: w * (1 - w * w)
check("notes: dw/dt = w(1 - w^2) is ~0.01 at 0.01, 0.375 at 0.5, 0 at 1", np.isclose(rate(.01), .01, atol=1e-5) and np.isclose(rate(.5), .375) and rate(1.) == 0)
def rk4(F, z0, T, n=20000):
    z, h, out = np.array(z0, float), T / n, [np.array(z0, float)]
    for _ in range(n):
        k1 = F(z); k2 = F(z + h / 2 * k1); k3 = F(z + h / 2 * k2); k4 = F(z + h * k3)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); out.append(z)
    return np.array(out)
Fab = lambda z: np.array([z[1] * (1 - z[0] * z[1]), z[0] * (1 - z[0] * z[1])])
Z = rk4(Fab, [.1, .1], 8, 16000); ts = np.arange(0, 8.0001, 1.); idx = (ts / 8 * 16000).astype(int)
uu = Z[idx, 0] * Z[idx, 1]; u0 = .01
check("notes: with a = b, u = ab follows u(t) = 1 / (1 + ((1 - u0)/u0) e^{-2t})", np.allclose(uu, 1 / (1 + (1 - u0) / u0 * np.exp(-2 * ts)), atol=1e-8))
check("notes: d/dw of (1 - w^2)^2 / 2 is twice the one-layer rate", np.isclose(-fd_grad(lambda w: .5 * (1 - w[0] ** 2) ** 2, np.array([.5]))[0], 2 * rate(.5)))
check("notes: orthogonal Q preserves length", np.isclose(np.linalg.norm(np.linalg.qr(np.random.default_rng(2).normal(size=(6, 6)))[0] @ np.arange(6.)), np.linalg.norm(np.arange(6.))))
# primer: saddle at the origin, conserved a^2 - b^2, mode half-time ln(s/u0 - 1)/(2s), depth products
check("primer: Hessian at the origin [[0, -1], [-1, 0]], eigenvalues -1, +1", np.allclose(fd_hess(Ls, np.zeros(2)), [[0, -1], [-1, 0]], atol=1e-4) and np.allclose(eig([[0, -1], [-1, 0]]), [-1, 1]))
Z2 = rk4(Fab, [.3, .1], 30, 30000)
check("primer: a^2 - b^2 stays 0.08 from (0.3, 0.1), ends on ab = 1", np.allclose(Z2[:, 0] ** 2 - Z2[:, 1] ** 2, .08, atol=1e-8) and abs(Z2[-1, 0] * Z2[-1, 1] - 1) < 1e-6)
def half_time(s, u0=1e-3, h=1e-5):
    u, t = u0, 0.
    while u < s / 2:
        k1 = 2 * u * (s - u); k2 = 2 * (u + h / 2 * k1) * (s - u - h / 2 * k1); k3 = 2 * (u + h / 2 * k2) * (s - u - h / 2 * k2); k4 = 2 * (u + h * k3) * (s - u - h * k3)
        u_new = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if u_new >= s / 2: return t + h * (s / 2 - u) / (u_new - u)
        u, t = u_new, t + h
for s, tv in [(3, 1.33), (1, 3.45)]:
    th, tf = half_time(s), np.log(s / 1e-3 - 1) / (2 * s)
    check(f"primer: mode s = {s} half-learned at t ~ {tv} (Eq. 11 with u_f = s/2)", np.isclose(th, tf, atol=1e-4) and np.isclose(tf, tv, atol=.006), f"simulated {th:.4f}, formula {tf:.4f}")
check("primer: 0.9^50 ~ 0.005, 1.1^50 ~ 117", np.isclose(.9 ** 50, .005, atol=2e-4) and np.isclose(1.1 ** 50, 117, atol=.5), f"{.9 ** 50:.5f}, {1.1 ** 50:.2f}")
done()
