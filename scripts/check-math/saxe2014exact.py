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
done()
