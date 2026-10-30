"""Li, Rakhlin, Jadbabaie 2023: convergence of Adam under relaxed assumptions.

Definition 3.2: f is (rho, L0, L_rho) smooth if |grad^2 f(x)| <= L0 + L_rho |grad f(x)|^rho a.e.; Assumption 2: 0 <= rho < 2.
rho = 1 is Zhang et al.'s (L0, L1) smoothness. Prop. B.1: rational functions and double exponentials are (rho, L0, L_rho)
smooth with 1 < rho < 2 but not necessarily (L0, L1) smooth.
Theorem 4.1: with high probability |grad f(x_t)| <= G along Adam's trajectory, and (1/T) sum |grad f(x_t)|^2 <= eps^2,
O(eps^-4) gradient complexity (no restriction on beta_sq).
3D toy: f(x, y) = cosh(1.5 x) + (y^2 - 1)^2 (not globally L-smooth, (rho, L0, L_rho) smooth); GD vs Adam, same step.
"""
from common import *

print("li2023convergence")
F = lambda z: np.cosh(1.5 * z[0]) + (z[1] ** 2 - 1) ** 2
G = lambda z: np.array([1.5 * np.sinh(1.5 * z[0]), 4 * z[1] * (z[1] ** 2 - 1)])
Hd = lambda z: np.array([2.25 * np.cosh(1.5 * z[0]), 12 * z[1] ** 2 - 4])
check_grad("cosh(1.5x) + (y^2 - 1)^2", F, G, rand_pts(6, -2, 2))
check_hess_at("at (1, 0.5)", F, [1, .5], np.diag(Hd([1, .5])))
# generalized smoothness of the toy, checked numerically on a box: |H| <= L0 + L1 |g|
P = rand_pts(4000, -3, 3, seed=1)
ratio = max(np.abs(Hd(p)).max() / (8 + 3 * np.linalg.norm(G(p))) for p in P)
check("toy is (1, L0, L1) smooth on [-3, 3]^2 with L0 = 8, L1 = 3", ratio <= 1, f"max |H| / (8 + 3|g|) = {ratio:.3f}")
check("toy is not globally L-smooth (|H| unbounded)", Hd([8, 0])[0] > 1e4, f"|H| at x = 8: {Hd([8, 0])[0]:.0f}")
# 1D examples for the second tab
xs = np.linspace(.2, 6, 200)
ex = (np.exp(xs), np.exp(xs))
check("e^x: f'' = f' (rho = 1, L0 = 0, L1 = 1)", np.allclose(*ex))
q1, q2 = 4 * xs ** 3, 12 * xs ** 2
check("x^4: f'' = 12 (|f'|/4)^(2/3) (rho = 2/3)", np.allclose(q2, 12 * (q1 / 4) ** (2 / 3)))
r1, r2 = 1 / xs ** 2, 2 / xs ** 3
check("1/x: |f''| = 2 |f'|^(3/2) (rho = 3/2, not (L0, L1))", np.allclose(r2, 2 * r1 ** 1.5))
x2 = np.linspace(0, 2.5, 50); d1 = np.exp(x2) * np.exp(np.exp(x2)); d2 = np.exp(np.exp(x2)) * np.exp(x2) * (1 + np.exp(x2))
check("e^(e^x): f''/f' = 1 + e^x grows without bound (not (L0, L1))", np.allclose(d2 / d1, 1 + np.exp(x2)) and (d2 / d1)[-1] > 10)
check("e^(e^x): f'' <= L0 + L_rho f'^rho with rho = 1.5", np.all(d2 <= 2 + 2 * d1 ** 1.5))

# 3D tab: GD vs Adam from (3.6, 1.9) with eta = 0.05
START, ETA = np.array([3.6, 1.9]), .05


def gd(T=400):
    z = START.copy(); out = [np.linalg.norm(G(z))]
    for _ in range(T):
        z = z - ETA * G(z)
        if not np.all(np.isfinite(z)) or np.abs(z).max() > 20: return out, False, z
        out.append(np.linalg.norm(G(z)))
    return out, True, z


def adam(T=1000, b1=.9, b2=.999, lam=1e-8):
    z = START.copy(); m = np.zeros(2); v = np.zeros(2); out = [np.linalg.norm(G(z))]
    for t in range(1, T + 1):
        g = G(z); m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        z = z - ETA * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + lam)
        out.append(np.linalg.norm(G(z)))
    return out, True, z


og, okg, zg = gd(); oa, _, za = adam()
check("GD with eta = 0.05 from (3.6, 1.9) diverges (each step overshoots further: curvature 2.25 cosh(5.4) ~ 250)", not okg, f"after {len(og)} steps")
check("Adam with the same eta converges near a minimum (0, +-1)", abs(za[0]) < .05 and abs(abs(za[1]) - 1) < .05, f"end {np.round(za, 3)}")
check("Adam's gradient norms stay bounded by their initial value", max(oa) <= oa[0] * 1.0001, f"max |g| = {max(oa):.2f}, start {oa[0]:.2f}")
gd_small = START.copy()
for t in range(1500):
    gd_small = gd_small - .01 * G(gd_small)
check("a smaller GD step (0.01) does converge from the same start", np.linalg.norm(G(gd_small)) < 1e-3)
save_json("li2023convergence", "data.json", {"adam": sig(oa, 4), "gd": sig(og, 4)})
done()
