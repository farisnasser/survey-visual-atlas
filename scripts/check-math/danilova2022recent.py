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

# in-page notes (papers/danilova2022recent/notes.md): GD baseline (Sec. 1), PL contraction (Sec. 3), primer numbers
fD = lambda p: p[0] ** 2 + 3 * np.sin(p[0]) ** 2 + .5 * p[1] ** 2
gD = lambda p: np.array([2 * p[0] + 3 * np.sin(2 * p[0]), p[1]])
LD = 8.
ok1, ok2 = True, True
for p0 in rand_pts(30, -5, 5, seed=24):
    p, Fs, Gs = p0.copy(), [], []
    for k in range(60):
        Fs.append(fD(p)); Gs.append(gD(p) @ gD(p)); pn = p - gD(p) / LD
        ok1 &= fD(pn) <= fD(p) - (gD(p) @ gD(p)) / (2 * LD) + 1e-12
        p = pn
    ok2 &= np.mean(Gs) <= 2 * LD * (Fs[0] - 0) / len(Gs) + 1e-12
check("notes: GD with step 1/L: f(x+) <= f(x) - |grad|^2/(2L)", ok1)
check("notes: (1/T) sum |grad|^2 <= 2L[f(x0) - f*]/T", ok2)
xs6 = np.linspace(-6, 6, 1200001); xs6 = xs6[np.abs(xs6) > 1e-4]
Fx = xs6 ** 2 + 3 * np.sin(xs6) ** 2; gx = 2 * xs6 + 3 * np.sin(2 * xs6); muD = (gx ** 2 / (2 * Fx)).min()
ok3 = True
xk = 5.
for k in range(40):
    Fk = xk ** 2 + 3 * np.sin(xk) ** 2; xn = xk - (2 * xk + 3 * np.sin(2 * xk)) / LD
    ok3 &= xn ** 2 + 3 * np.sin(xn) ** 2 <= (1 - muD / LD) * Fk + 1e-15; xk = xn
check("notes: under PL, f(x+) - f* <= (1 - mu/L)(f(x) - f*) along GD", ok3)
# primer
check("primer: x^2/2 at 0.1: gradient 0.1, gap 0.005", abs(.1 - .1) < 1e-15 and abs(.5 * .1 ** 2 - .005) < 1e-15)
check("primer: eps = 1e-2: eps^-2 = 1e4, eps^-4 = 1e8", abs(1e-2 ** -2 - 1e4) < 1e-6 and abs(1e-2 ** -4 - 1e8) < 1e-3)
check("primer: x^2 + 3 sin^2 x: f'' min -4, max L = 8, PL mu ~ 0.176 on [-6, 6]", abs((2 + 6 * np.cos(2 * xs6)).min() + 4) < 1e-9 and abs((2 + 6 * np.cos(2 * xs6)).max() - 8) < 1e-9 and abs(muD - .176) < 5e-4, f"{muD:.4f}")
kb = np.log(1e6) / -np.log(1 - muD / LD)
xk, F0, kk = 5., 5. ** 2 + 3 * np.sin(5.) ** 2, None
for k in range(700):
    if xk ** 2 + 3 * np.sin(xk) ** 2 <= 1e-6 * F0:
        kk = k; break
    xk -= (2 * xk + 3 * np.sin(2 * xk)) / LD
check("primer: guarantee ~623 steps for a 1e6 reduction; GD from x0 = 5 needs 10", abs(kb - 623) < 1 and kk == 10, f"{kb:.1f}, {kk}")
check("primer: forward difference of x^2 at 1 with delta 1e-3 = 2.001", abs(((1 + 1e-3) ** 2 - 1) / 1e-3 - 2.001) < 1e-9)
done()
