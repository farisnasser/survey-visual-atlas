"""Crash course stop 2 (course/slope): derivatives, gradient compass, tangent plane, Jacobian, chain rule."""
from common import *

print("course_slope")
f = lambda x: x * x
check("text: secant of x^2 is 2x + h", all(np.isclose((f(x + h) - f(x)) / h, 2 * x + h) for x in (-1.3, .4, 2) for h in (.1, .01)))
check_grad("bowl x^2 + 3y^2", lambda p: p[0] ** 2 + 3 * p[1] ** 2, lambda p: np.array([2 * p[0], 6 * p[1]]), rand_pts(20, -2, 2))
check_grad("saddle x^2 - y^2", lambda p: p[0] ** 2 - p[1] ** 2, lambda p: np.array([2 * p[0], -2 * p[1]]), rand_pts(20, -2, 2))
# directional derivative = grad . u, maximal along the gradient
q = np.array([1.1, .55]); gq = np.array([2 * q[0], 6 * q[1]]); fb = lambda p: p[0] ** 2 + 3 * p[1] ** 2
angs = np.linspace(0, 2 * np.pi, 3601); dd = [(fb(q + 1e-6 * np.array([np.cos(a), np.sin(a)])) - fb(q - 1e-6 * np.array([np.cos(a), np.sin(a)]))) / 2e-6 for a in angs]
check("D_u f = grad . u for every direction", np.allclose(dd, [gq @ [np.cos(a), np.sin(a)] for a in angs], atol=1e-5))
check("max over u of D_u f is |grad f|, attained along grad f", np.isclose(max(dd), np.linalg.norm(gq), atol=1e-5) and abs(angs[int(np.argmax(dd))] - np.arctan2(gq[1], gq[0]) % (2 * np.pi)) < 2e-3)
# the gradient is perpendicular to the contour: tangent of the level set (-g_y, g_x) has zero directional derivative
tang = np.array([-gq[1], gq[0]]) / np.linalg.norm(gq)
check("gradient is perpendicular to the contour", abs(gq @ tang) < 1e-12)

fp = lambda p: .3 * p[0] ** 2 + .2 * p[1] ** 2 + .4 * np.sin(1.5 * p[0]) * np.cos(p[1])
gp = lambda p: np.array([.6 * p[0] + .6 * np.cos(1.5 * p[0]) * np.cos(p[1]), .4 * p[1] - .4 * np.sin(1.5 * p[0]) * np.sin(p[1])])
check_grad("tangent-plane surface", fp, gp, rand_pts(30, -2.4, 2.4))
x0 = np.array([.7, -.4]); u = gp(x0) / np.linalg.norm(gp(x0))
errs = [abs(fp(x0 + r * u) - fp(x0) - gp(x0) @ (r * u)) for r in (.4, .2, .1, .05)]
check("Taylor error shrinks like |Delta|^2 (ratio ~4 per halving)", all(3 < a / b < 5 for a, b in zip(errs, errs[1:])), f"errors {np.round(errs, 6).tolist()}")

G = lambda p: np.array([p[0] + .5 * np.sin(p[1]), p[1] + .5 * np.sin(p[0])])
J = lambda p: np.array([[1, .5 * np.cos(p[1])], [.5 * np.cos(p[0]), 1]])
for p in rand_pts(10, -2, 2, seed=4):
    Jfd = np.column_stack([(G(p + e) - G(p - e)) / 2e-6 for e in np.eye(2) * 1e-6])
    assert np.allclose(J(p), Jfd, atol=1e-6)
check("Jacobian of g(x, y) = (x + 0.5 sin y, y + 0.5 sin x)", True)
check("det J = 1 - 0.25 cos x cos y > 0 everywhere", all(np.isclose(np.linalg.det(J(p)), 1 - .25 * np.cos(p[0]) * np.cos(p[1])) for p in rand_pts(20, -3, 3)) and 1 - .25 > 0)
p = np.array([.9, -.6]); gaps = []
for s in (.8, .4, .2, .1):
    h = s / 2; gaps.append(max(np.linalg.norm(G(p + [a, b]) - G(p) - J(p) @ [a, b]) for a in (-h, h) for b in (-h, h)))
check("visual 'jacobian': corner gap shrinks like size^2", all(3 < a / b < 5 for a, b in zip(gaps, gaps[1:])), f"gaps {np.round(gaps, 6).tolist()}")
# chain rule example: f(w) = (2w - 1)^2, f'(w) = 4(2w - 1), f'(1) = 4
fw = lambda w: (2 * w - 1) ** 2
check("chain rule example f'(1) = 4", np.isclose((fw(1 + 1e-6) - fw(1 - 1e-6)) / 2e-6, 4))
# secant visual function = stop 1 loss
F = lambda t: .1 * t * t + .5 * np.sin(3 * t); dF = lambda t: .2 * t + 1.5 * np.cos(3 * t)
check("secant slope -> f'(x) as h -> 0", abs((F(-1.2 + 1e-3) - F(-1.2)) / 1e-3 - dF(-1.2)) < 3e-3 and abs((F(-1.2 + .1) - F(-1.2)) / .1 - dF(-1.2)) > .05)
done()
