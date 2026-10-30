"""Du, Jin, Lee, Jordan, Poczos, Singh 2017 (NeurIPS; arXiv 1705.10412): GD can take exponential time to escape saddle points.
Theorem 4.1: f on R^d, B-bounded, l-gradient- and rho-Hessian-Lipschitz (poly(d)), init U[-1,1]^d: GD (eta <= 1/l) stays Omega(1)
away from local minima for T <= e^Omega(d), while perturbed GD finds one in poly(d, 1/eps). Mechanism (Sec. 4, Remark): GD visits d
saddles in turn; the time to escape the i-th grows by a factor (L + gamma)/gamma each time; PGD escapes each in ~1/(eta gamma).

Our smooth chain (not the paper's "octopus" spline construction, same mechanism):
    f(x) = -g x1^2/2 + sum_{i>=2} q(x_{i-1}) x_i^2 / 2 + sum_i x_i^4 / 4,   q(t) = L - (L + g) S(t^2 / c^2),
S the quintic smoothstep (S(0) = 0, S(u) = 1 for u >= 1). While x_{i-1} ~ 0, coordinate i has curvature +L and SHRINKS; once x_{i-1}
has escaped (|x_{i-1}| >= c) its curvature is -g and it must escape in turn: a chain of strict saddles, minima at x = (+-1, ..., +-1).
"""
from common import *

print("du2017gradient")
G_, CC = 1., .5
S = lambda u: np.where(u >= 1, 1., 6 * u ** 5 - 15 * u ** 4 + 10 * u ** 3) * (u > 0)
dS = lambda u: np.where((u > 0) & (u < 1), 30 * u ** 4 - 60 * u ** 3 + 30 * u ** 2, 0.)
def make(L, d):
    q = lambda t: L - (L + G_) * S(t ** 2 / CC ** 2)
    dq = lambda t: -(L + G_) * dS(t ** 2 / CC ** 2) * 2 * t / CC ** 2
    def f(x):
        x = np.asarray(x, float); return -G_ * x[..., 0] ** 2 / 2 + np.sum(q(x[..., :-1]) * x[..., 1:] ** 2 / 2, -1) + np.sum(x ** 4 / 4, -1)
    def g(x):
        x = np.asarray(x, float); out = x ** 3; out[..., 0] += -G_ * x[..., 0]
        out[..., 1:] += q(x[..., :-1]) * x[..., 1:]; out[..., :-1] += dq(x[..., :-1]) * x[..., 1:] ** 2 / 2
        return out
    return f, g
f2, g2 = make(1., 2)
check_grad("chain, d = 2, L = 1", f2, g2, rand_pts(30, -1.4, 1.4))
f5, g5 = make(2., 5)
check_grad("chain, d = 5, L = 2", f5, g5, rand_pts(30, -1.4, 1.4, dim=5))
check_hess_at("d = 2 at 0: diag(-g, L) (first saddle)", f2, [0, 0], [[-1, 0], [0, 1]])
check_hess_at("d = 2 at (1, 0): diag(2g, -g) (second saddle)", f2, [1, 0], [[2, 0], [0, -1]])
check("d = 2: minima at (+-1, +-1), gradient 0, f = -1/2", all(np.linalg.norm(g2([a, b])) < 1e-12 and abs(f2([a, b]) + .5) < 1e-12 for a in (1, -1) for b in (1, -1)))
G = np.linspace(-1.6, 1.6, 321); P = np.stack(np.meshgrid(G, G), -1)
check("d = 2: nothing below -1/2 on the box", f2(P).min() >= -.5 - 1e-12)

def escapes(L, d, T, seed, pgd=False, eta=None, r=1e-2, x0=None):
    """first step at which |x_k| > c for each k (0-based list), GD or perturbed GD (kick of radius r when |grad| < r)."""
    f, g = make(L, d); eta = eta or 1 / (4 * (L + 3)); R = np.random.default_rng(seed)
    x = R.uniform(-.1, .1, d) if x0 is None else np.array(x0, float); out = []
    for t in range(T):
        while len(out) < d and abs(x[len(out)]) > CC: out.append(t)       # saddles are cleared in order: x_1 first, then x_2, ...
        if len(out) == d: break
        if pgd and np.linalg.norm(g(x)) < r: u = R.normal(size=d); x = x + r * R.uniform() ** (1 / d) * u / np.linalg.norm(u)
        x = x - eta * g(x)
    return out + [None] * (d - len(out)), x
# the 3D tab: two-saddle chain from (0.001, 0.05)
two = {L: escapes(L, 2, 8000, 0, x0=[.001, .05])[0] for L in [1., 2., 4.]}
print(f"  d = 2 from (0.001, 0.05): GD leaves saddle 1 / saddle 2 at {two}")
check("d = 2: the second wait grows with L (GD)", two[1.][1] - two[1.][0] < two[2.][1] - two[2.][0] < two[4.][1] - two[4.][0])
# d = 6 keeps the shrinking coordinates above double-precision underflow (they fall to ~ e^-(eta L T)); much larger d would
# make them exactly 0 and GD would then never escape in floating point.
for L in [1., 2.]:
    e, _x = escapes(L, 6, 400000, 0); ep, _ = escapes(L, 6, 400000, 0, pgd=True)
    ee = [q for q in e if q is not None]; gaps = np.diff([0] + ee); ratios = gaps[2:] / gaps[1:-1]
    print(f"  L = {L}, d = 6, x0 ~ U[-0.1,0.1]^6: GD escape steps {e}; gap ratios {np.round(ratios, 2).tolist()} (paper: (L+g)/g = {(L + G_) / G_})")
    print(f"           PGD (r = 0.01) escape steps {ep}; GD final point {np.array2string(_x, precision=3)}")
    check(f"L = {L}: GD's time per saddle grows geometrically, ratio ~ (L+g)/g (within 25% for the later saddles)", np.all(np.abs(ratios[-3:] / ((L + G_) / G_) - 1) < .25))
    check(f"L = {L}: PGD's time per saddle stays bounded (max gap < 3x the first gap) and PGD finishes far sooner", max(np.diff([0] + ep)[1:]) < 3 * max(np.diff([0] + ep)[1], 1) + 400 and (e[-1] is None or ep[-1] < e[-1] / 4))
# offline data for the 2D tab: escape steps per saddle, d = 6, five random starts each
data = {}
for L in [1., 2.]:
    data[str(L)] = {"gd": [escapes(L, 6, 400000, sd)[0] for sd in range(5)], "pgd": [escapes(L, 6, 400000, sd, pgd=True)[0] for sd in range(5)]}
    print(f"  d = 6, L = {L}: GD last escape {[q[-1] for q in data[str(L)]['gd']]}, PGD last escape {[q[-1] for q in data[str(L)]['pgd']]}")
check("PGD clears all 6 saddles in every run", all(b[-1] is not None for L in data for b in data[L]["pgd"]))
check("d = 6, L = 2: GD needs > 5x the steps of PGD, or never clears the last saddle within 400 000 steps", all(a[-1] is None or a[-1] > 5 * b[-1] for a, b in zip(data["2.0"]["gd"], data["2.0"]["pgd"])))
save_json("du2017gradient", "data.json", {"esc": data, "g": G_, "c": CC})
done()
