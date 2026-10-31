"""Crash course stop 3 (course/steps): step size on a parabola, Lipschitz, descent lemma, O(eps^-2)."""
from common import *

print("course_steps")
# parabola a = 4, x0 = 1
a = 4
check("eta = 0.1: factor 0.6, x10 = 0.006", np.isclose(1 - .1 * a, .6) and round(.6 ** 10, 3) == .006)
check("eta = 0.4: factor -0.6", np.isclose(1 - .4 * a, -.6))
check("eta = 0.6: factor -1.4, x10 = 28.9", np.isclose(1 - .6 * a, -1.4) and round((-1.4) ** 10, 1) == 28.9)
for eta in (.1, .25, .4, .5, .6):
    x = gd(lambda v: a * v, [1.0], eta, 10)[-1][0]
    assert np.isclose(x, (1 - eta * a) ** 10)
check("GD iterates on the parabola are (1 - eta a)^t x0", True)
# bowl 0.5(x^2 + 10 y^2)
bL = lambda p: .5 * (p[0] ** 2 + 10 * p[1] ** 2); bg = lambda p: np.array([p[0], 10 * p[1]])
check_grad("elongated bowl", bL, bg, rand_pts(20, -1.6, 1.6))
check("bowl: eta = 0.17 converges, 0.21 diverges (threshold 0.2)", bL(gd(bg, [-1.45, .5], .17, 400)[-1]) < 1e-8 and abs(gd(bg, [-1.45, .5], .21, 60)[-1][1]) > .7)
# Lipschitz example f = 0.8 sin 2x + 0.3x
xs = np.linspace(-10, 10, 2000001)
check("max |f'| of 0.8 sin 2x + 0.3x is 1.9", np.isclose(np.max(np.abs(1.6 * np.cos(2 * xs) + .3)), 1.9, atol=1e-9))
# smoothness example f = 0.4 cos 2x + 0.1 x^2
FS = lambda x: .4 * np.cos(2 * x) + .1 * x * x; dFS = lambda x: -.8 * np.sin(2 * x) + .2 * x; d2 = -1.6 * np.cos(2 * xs) + .2
check("f'' ranges over [-1.4, 1.8] so L = 1.8", np.isclose(d2.min(), -1.4, atol=1e-9) and np.isclose(d2.max(), 1.8, atol=1e-9))
fstar = FS(xs).min(); xstar = abs(xs[np.argmin(FS(xs))])
print(f"  f* = {fstar:.6f} at |x| = {xstar:.4f}")
check("f* = -0.180931 (viz constant)", np.isclose(fstar, -.180931, atol=2e-6))
Ls = 1.8
ok = all(FS(y) <= FS(x) + dFS(x) * (y - x) + Ls / 2 * (y - x) ** 2 + 1e-12 for x in np.linspace(-3, 3, 121) for y in np.linspace(-4, 4, 161))
check("quadratic upper bound holds with L = 1.8", ok)
bad = any(FS(y) > FS(x) + dFS(x) * (y - x) + 1.2 / 2 * (y - x) ** 2 + 1e-9 for x in np.linspace(1.37, 1.77, 21) for y in np.linspace(.8, 2.4, 81))
check("and fails for L = 1.2 near x = 1.57, where f'' = 1.8 (visual hint)", bad and np.isclose(-1.6 * np.cos(2 * np.pi / 2) + .2, 1.8))
check("descent lemma: f(x - f'(x)/L) <= f(x) - f'(x)^2/(2L)", all(FS(x - dFS(x) / Ls) <= FS(x) - dFS(x) ** 2 / (2 * Ls) + 1e-12 for x in np.linspace(-3, 3, 601)))
check("text: T = 2 L (f0 - f*) / eps^2 = 20000 for L = 1, gap 1, eps = 0.01", np.isclose(2 * 1 * 1 / .01 ** 2, 2e4))
for x0 in (.002, .5, -2.5, 2.8):
    x, m, ok = x0, np.inf, True
    for T in range(1, 1001):
        m = min(m, abs(dFS(x))); ok &= m <= np.sqrt(2 * Ls * (FS(x0) - fstar) / T) + 1e-12; x -= dFS(x) / Ls
    assert ok, x0
check("visual 'rate': min gradient stays below sqrt(2L(f0 - f*)/T) for several starts", True)
done()
