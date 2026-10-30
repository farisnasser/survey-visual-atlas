"""Danilova et al. 2022 ("Recent theoretical advances in non-convex optimization", Springer volume; arXiv 2012.06188), Sec. 5.1:
Polyak-Lojasiewicz condition f(x) - f* <= |grad f(x)|^2 / (2 mu) (Eq. 36): every stationary point is global, and for L-smooth f, GD with
step 1/L converges linearly, f(x_k) - f* <= exp(-mu k / L) (f(x_0) - f*).
Toy: f(x, y) = x^2 + 3 sin^2 x + y^2 (the 1-D part is the classic nonconvex PL example of Karimi et al. 2016): nonconvex, PL, L = 8.
"""
from common import *

print("danilova2022recent")
f = lambda p: p[..., 0] ** 2 + 3 * np.sin(p[..., 0]) ** 2 + p[..., 1] ** 2
g = lambda p: np.stack([2 * p[..., 0] + 3 * np.sin(2 * p[..., 0]), 2 * p[..., 1]], -1)
check_grad("f", lambda p: float(f(np.asarray(p))), lambda p: g(np.asarray(p)), rand_pts(20, -4, 4))
xs = np.linspace(-6, 6, 200001); fxx = 2 + 6 * np.cos(2 * xs)
L = 8.
check("nonconvex: f_xx = 2 + 6 cos 2x reaches -4", fxx.min() < -3.99)
check("L-smooth with L = 8", fxx.max() <= L + 1e-12)
X = np.stack([xs, np.zeros_like(xs)], 1); m = np.abs(xs) > 1e-4
ratio = np.sum(g(X) ** 2, 1)[m] / (2 * f(X)[m])
mu1 = float(ratio.min()); mu = min(mu1, 2.)          # the y-part gives (2y)^2 / (2 y^2) = 2
print(f"  PL constant of the x-part on [-6, 6]: min |f'|^2 / (2 (f - f*)) = {mu1:.4f} at x = {xs[m][np.argmin(ratio)]:.3f}; mu = {mu:.4f} (Karimi et al. quote 1/32 = {1 / 32:.4f})")
check("PL holds with mu >= 1/32 (the only stationary point is the global minimum 0)", mu >= 1 / 32)
R = np.random.default_rng(0); P0 = np.stack([R.uniform(-5, 5, 200), R.uniform(-5, 5, 200)], 1)
P = P0.copy(); gaps = [float(np.max(f(P)))]; ok = True
for k in range(1, 401):
    P = P - g(P) / L; gaps.append(float(np.max(f(P)))); ok &= np.all(f(P) <= np.exp(-mu * k / L) * f(P0) + 1e-15)
print(f"  GD (step 1/L) from 200 starts in [-5, 5]^2: max f - f* after 50, 100, 400 steps: {gaps[50]:.2e}, {gaps[100]:.2e}, {gaps[400]:.2e}")
check("every run satisfies f(x_k) - f* <= exp(-mu k / L) (f(x_0) - f*) for 400 steps", ok)
check("all 200 runs reach the global minimum despite non-convexity", gaps[400] < 1e-12)
ks = list(range(0, 401, 5))
save_json("danilova2022recent", "data.json", {"mu": mu, "L": L, "gaps": [[k, float(f"{gaps[k]:.4g}")] for k in ks], "f0max": float(np.max(f(P0)))})
done()
