"""Nesterov 2004 (Introductory Lectures), as used by the survey: L-smoothness, the descent lemma and the gradient-method bound.

f(x, y) = x^2/4 - cos(2y)/4: Hessian diag(1/2, cos 2y), so grad f is L-Lipschitz with L = 1 exactly; non-convex where
cos 2y < 0; minima (0, k pi) with f* = -1/4, saddles (0, pi/2 + k pi).
Descent lemma: f(y) <= f(x) + <grad f(x), y - x> + L/2 |y - x|^2  =>  one gradient step with step h gives
f(x - h grad) <= f(x) - h (1 - L h / 2) |grad|^2  (decrease for 0 < h < 2/L, best h = 1/L: f drops by |grad|^2 / (2L)).
Summing: min_{i <= k} |grad f(x_i)|^2 <= 2 L (f(x_0) - f*) / (k + 1)  for h = 1/L.
"""
from common import *

print("nesterov2004introductory")
f = lambda p: p[0] ** 2 / 4 - np.cos(2 * p[1]) / 4
g = lambda p: np.array([p[0] / 2, np.sin(2 * p[1]) / 2])
check_grad("f", f, g, rand_pts(20, -3, 3))
check_hess_at("at (0.3, 1.0)", f, np.array([.3, 1.0]), np.diag([.5, np.cos(2.0)]))
L = 1.0
R = np.random.default_rng(44)
P = rand_pts(5000, -4, 4, seed=1); Q = rand_pts(5000, -4, 4, seed=2)
lip = max(np.linalg.norm(g(p) - g(q)) / np.linalg.norm(p - q) for p, q in zip(P, Q))
check("grad f is 1-Lipschitz (sampled ratio <= 1)", lip <= 1 + 1e-12, f"max ratio {lip:.4f}")
dl = max(f(q) - f(p) - g(p) @ (q - p) - L / 2 * np.sum((q - p) ** 2) for p, q in zip(P, Q))
check("descent lemma: f(y) <= f(x) + <grad, y - x> + L/2 |y - x|^2", dl <= 1e-12, f"max violation {dl:.2e}")
for h in (.5, 1.0, 1.9):
    worst = max(f(p - h * g(p)) - (f(p) - h * (1 - L * h / 2) * np.sum(g(p) ** 2)) for p in P)
    check(f"one-step decrease bound holds for h = {h}/L", worst <= 1e-12)
x0 = np.array([1.8, 1.3]); fstar = -.25
path = gd(g, x0, 1 / L, 400)
G2 = np.array([np.sum(g(p) ** 2) for p in path])
mins = np.minimum.accumulate(G2)
bound = 2 * L * (f(x0) - fstar) / (np.arange(len(G2)) + 1)
check("min_{i<=k} |grad|^2 <= 2L (f(x0) - f*)/(k + 1) along the h = 1/L run", np.all(mins <= bound + 1e-15))
check("h = 1/L run is monotone and ends at a minimum", np.all(np.diff([f(p) for p in path]) <= 1e-15) and abs(f(path[-1]) - fstar) < 1e-8)
p22 = gd(g, x0, 2.2 / L, 30)
check("h = 2.2/L > 2/L: f increases at some step (no descent guarantee; near a minimum the y-direction factor |1 - 2.2| > 1)",
      any(f(p22[i + 1]) > f(p22[i]) for i in range(29)))
save_json("nesterov2004introductory", "data.json", {"L": L, "fstar": fstar, "x0": x0.tolist(), "G2": sig(np.maximum(G2, 1e-30)[:200], 4)})
# in-page notes (papers/nesterov2004introductory/notes.md): Sec. 2 descent lemma and step, Sec. 3 telescoped bound, Sec. 4 x^3
fn = lambda p: p[0] ** 2 / 4 - np.cos(2 * p[1]) / 4
gn = lambda p: np.array([p[0] / 2, np.sin(2 * p[1]) / 2])
check_grad("primer/notes: f = x^2/4 - cos(2y)/4", fn, gn, rand_pts(20, -3, 3))
Rn4 = np.random.default_rng(9)
ok = True
for _ in range(500):
    p, s = Rn4.uniform(-3, 3, 2), Rn4.normal(size=2) * Rn4.uniform(0, 2)
    ok &= fn(p + s) <= fn(p) + gn(p) @ s + .5 * s @ s + 1e-12
    e = Rn4.uniform(0, 2)
    ok &= fn(p - e * gn(p)) <= fn(p) - e * (1 - e / 2) * gn(p) @ gn(p) + 1e-12
check("notes: descent lemma and f(x - eta g) <= f(x) - eta(1 - L eta/2)|g|^2 with L = 1 (500 random cases)", ok)
p = np.array([3., 1.]); D = fn(p) + .25; gsq = []
for t in range(20):
    gsq.append(gn(p) @ gn(p)); p1 = p - gn(p)
    assert fn(p1) <= fn(p) - .5 * gsq[-1] + 1e-12
    p = p1
check("notes: eta = 1/L gives f(x_{t+1}) <= f(x_t) - |g|^2/(2L), and min |g|^2 <= 2 L Delta / T", min(gsq) <= 2 * D / 20 and np.sum(gsq) / 2 <= D)
check("notes: x^3 has f'(0) = 0 but 0 is not a local minimum", np.isclose(fd_grad(lambda q: q[0] ** 3, np.array([0.]))[0], 0, atol=1e-9) and (-1e-3) ** 3 < 0)
# primer: L = 1 for the example, telescoping, counting, the run
check("primer: Hessian diag(1/2, cos 2y): eigenvalues within [-1, 1], cos 2y < 0 somewhere", all(np.max(np.abs(eig(fd_hess(fn, np.array([.3, y]))))) <= 1 + 1e-3 for y in np.linspace(-2, 2, 21)) and np.cos(2 * 1.) < 0)
check("primer: decreases 3 + 2 + 0.5 = 10 - 4.5 = 5.5", np.isclose(np.sum(-np.diff([10, 7, 5, 4.5])), 5.5))
check("primer: L = 1, Delta = 10, eps = 0.01 -> T = 200,000", np.isclose(2 * 1 * 10 / .01 ** 2, 200000))
check("primer: from (3, 1): Delta ~ 2.60, bound after 20 steps ~ 0.26, run reaches ~1e-11", np.isclose(D, 2.60, atol=.005) and np.isclose(2 * D / 20, .26, atol=.005) and 1e-12 < min(gsq) < 1e-10, f"{D:.4f}, {min(gsq):.2e}")
done()
