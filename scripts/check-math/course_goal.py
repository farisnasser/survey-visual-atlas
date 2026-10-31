"""Crash course stop 1 (course/goal): line-fit landscape and the three-valley loss."""
from common import *

print("course_goal")
X = np.array([-1.5, -1, -.5, 0, .5, 1, 1.5, 2]); Y = np.array([-.95, -.42, -.18, .41, .62, 1.18, 1.31, 1.86])
L = lambda p: np.mean((p[0] * X + p[1] - Y) ** 2)
g = lambda p: np.array([np.mean(2 * (p[0] * X + p[1] - Y) * X), np.mean(2 * (p[0] * X + p[1] - Y))])
check_grad("line-fit loss", L, g, rand_pts(30, -1, 2))
w, b = np.linalg.lstsq(np.c_[X, np.ones_like(X)], Y, rcond=None)[0]
print(f"  w* = {w:.6f}, b* = {b:.6f}, L* = {L([w, b]):.6f}, L(0,0) = {L([0, 0]):.6f}, L(1,0) = {L([1, 0]):.6f}")
check("text: w* = 0.776, b* = 0.285, L* = 0.0086", round(w, 3) == .776 and round(b, 3) == .285 and round(L([w, b]), 4) == .0086)
check("text: L(0,0) = 1.03, L(1,0) = 0.126", round(L([0, 0]), 2) == 1.03 and round(L([1, 0]), 3) == .126)
H = 2 / len(X) * np.array([[np.sum(X * X), np.sum(X)], [np.sum(X), len(X)]])
check_hess_at("line-fit loss (constant)", L, [.3, .2], H)
lam = eig(H)
check("visual 'fit': GD step 0.15 is stable (0.15 < 2/lambda_max)", .15 < 2 / lam[-1], f"lambda = {np.round(lam, 3).tolist()}")
p = gd(g, [0, 0], .15, 400)[-1]
check("visual 'fit': rolling from (0,0) reaches (w*, b*)", np.allclose(p, [w, b], atol=1e-6))

F = lambda t: .1 * t * t + .5 * np.sin(3 * t); dF = lambda t: .2 * t + 1.5 * np.cos(3 * t); d2F = lambda t: .2 - 4.5 * np.sin(3 * t)
check_grad("valleys loss", lambda v: F(v[0]), lambda v: np.array([dF(v[0])]), rand_pts(30, -3, 3, dim=1))
ts = np.linspace(-3, 3, 600001); v = F(ts)
mins = []
for i in np.where((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0] + 1:
    t = ts[i]
    for _ in range(50):
        t -= dF(t) / d2F(t)
    mins.append(t)
print("  minima:", [(round(m, 5), round(F(m), 5)) for m in mins])
check("three local minima at -2.50, -0.50, 1.50", len(mins) == 3 and np.allclose(np.round(mins, 2), [-2.50, -.50, 1.50]))
check("their losses 0.156, -0.474, -0.264", np.allclose(np.round([F(m) for m in mins], 3), [.156, -.474, -.264]))
check("the middle one is global on [-3, 3]", np.argmin([F(m) for m in mins]) == 1 and F(mins[1]) <= v.min() + 1e-9)
check("viz.js MINIMA constants match", np.allclose(mins, [-2.5045, -.5013, 1.50352], atol=1e-4))
check("visual 'valleys': GD step 0.05 is stable (max |F''| = 4.7)", .05 < 2 / 4.7)
ends = {}
for t0 in np.linspace(-3, 3, 61):
    t = t0
    for _ in range(4000):
        if abs(dF(t)) < 1e-4:
            break
        t -= .05 * dF(t)
    ends[round(t0, 2)] = mins[int(np.argmin([abs(t - m) for m in mins]))]
check("default start 2.6 rolls into the local minimum at 1.50", np.isclose(ends[2.6], mins[2]))
done()
