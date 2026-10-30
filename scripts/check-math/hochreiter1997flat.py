"""Hochreiter & Schmidhuber 1997 (Neural Computation 9(1)): a flat minimum is a large connected region in weight space where the
error stays approximately constant; by an MDL / Gibbs argument such weights need fewer bits (low precision suffices) and overfit
less; the algorithm (flat minimum search, FMS) looks for them. Toy: two minima of (almost) equal depth, one wide and one narrow:
    f(x, y) = 0.02 (x^2 + y^2) - exp(-|w - (-1, 0)|^2 / (2 0.55^2)) - exp(-|w - (1, 0)|^2 / (2 0.12^2))
"""
from common import *

print("hochreiter1997flat")
SF, SS = .55, .12
def f(p):
    p = np.asarray(p, float); x, y = p[..., 0], p[..., 1]
    return .02 * (x * x + y * y) - np.exp(-((x + 1) ** 2 + y ** 2) / (2 * SF ** 2)) - np.exp(-((x - 1) ** 2 + y ** 2) / (2 * SS ** 2))
def g(p):
    x, y = p; a = np.exp(-((x + 1) ** 2 + y ** 2) / (2 * SF ** 2)) / SF ** 2; b = np.exp(-((x - 1) ** 2 + y ** 2) / (2 * SS ** 2)) / SS ** 2
    return np.array([.04 * x + a * (x + 1) + b * (x - 1), .04 * y + a * y + b * y])
check_grad("f", f, g, rand_pts(30, -2, 2))
def newton(p):
    p = np.array(p, float)
    for _ in range(50): p = p - np.linalg.solve(fd_hess(f, p), g(p))
    return p
mF, mS = newton([-1, 0]), newton([1, 0])
fF, fS = float(f(mF)), float(f(mS))
check("two minima of almost equal depth (|difference| < 0.01)", abs(fF - fS) < .01, f"flat {mF.round(4)} f = {fF:.4f}, sharp {mS.round(4)} f = {fS:.4f}")
hF, hS = eig(fd_hess(f, mF)), eig(fd_hess(f, mS))
print(f"  Hessian eigenvalues: flat {hF.round(3)}, sharp {hS.round(3)}")
check("the narrow minimum is much sharper (curvature ratio > 15)", hS[0] / hF[1] > 15)
# epsilon-flat regions: area of the connected set {f < f(min) + eps} around each minimum (grid, flood from the minimum)
G = np.linspace(-2.6, 2.6, 1041); h = G[1] - G[0]; XX, YY = np.meshgrid(G, G); Z = f(np.stack([XX, YY], -1))
def area(m, fm, eps):
    mask = Z < fm + eps; i0, j0 = np.argmin(np.abs(G - m[1])), np.argmin(np.abs(G - m[0]))
    seen = np.zeros_like(mask); stack = [(i0, j0)]; n = 0
    while stack:
        i, j = stack.pop()
        if i < 0 or j < 0 or i >= len(G) or j >= len(G) or seen[i, j] or not mask[i, j]: continue
        seen[i, j] = True; n += 1; stack += [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
    return n * h * h
rows = []
for eps in [.02, .05, .1, .2]:
    aF, aS = area(mF, fF, eps), area(mS, fS, eps); rows.append([eps, aF, aS, float(np.log2(aF / aS))])
    print(f"  eps = {eps}: area flat {aF:.4f}, sharp {aS:.5f}, ratio {aF / aS:.1f} -> {np.log2(aF / aS):.1f} fewer bits to pin down the flat weights (2 weights)")
check("the flat region is > 15x larger at every eps (fewer bits of precision needed)", all(r_[1] / r_[2] > 15 for r_ in rows))
# robustness: expected loss increase under Gaussian weight noise of size s
R = np.random.default_rng(0); E = R.normal(size=(20000, 2)); rob = []
for s in [.02, .05, .1, .15, .2, .3]:
    rob.append([s, float(np.mean(f(mF + s * E)) - fF), float(np.mean(f(mS + s * E)) - fS)])
    print(f"  noise s = {s}: mean loss increase flat {rob[-1][1]:.4f}, sharp {rob[-1][2]:.4f}")
check("the sharp minimum loses more than 3x as much under the same weight noise", all(q[2] > 3 * q[1] for q in rob))
save_json("hochreiter1997flat", "data.json", {"mF": r(mF, 5), "mS": r(mS, 5), "fF": fF, "fS": fS, "areas": rows, "rob": rob})
done()
