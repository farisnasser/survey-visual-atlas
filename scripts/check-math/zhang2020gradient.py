"""Zhang, He, Sra, Jadbabaie 2020: why gradient clipping accelerates training.

Definition 1: (L0, L1)-smooth if |grad^2 f(x)| <= L0 + L1 |grad f(x)|. Lemma 2: polynomials of degree >= 3 are (L0, L1)-smooth
but not L-smooth. Clipped GD: x <- x - h grad f, h = min(eta_c, gamma eta_c / |grad f|). Theorem 3: clipped GD needs
O(L0 Delta / eps^2 + L1^2 Delta / L0) iterations. Theorem 4: fixed-step GD needs Omega(L1 M Delta / (eps^2 log M)),
M = sup |grad f| on the initial sublevel set; Theorem 6: O((M L1 + L0) Delta / eps^2) suffices. So clipping can be
arbitrarily faster when the initialization is poor (large M).
Toy: f(x, y) = x^2/2 + x^4/4 + y^2/2 (degree 4, Lemma 2).
"""
from common import *

print("zhang2020gradient")
F = lambda z: z[0] ** 2 / 2 + z[0] ** 4 / 4 + z[1] ** 2 / 2
G = lambda z: np.array([z[0] + z[0] ** 3, z[1]])
check_grad("x^2/2 + x^4/4 + y^2/2", F, G, rand_pts(5, -3, 3))
P = rand_pts(5000, -6, 6, seed=2)
ratio = max(max(1 + 3 * p[0] ** 2, 1) / (1 + 3 * np.linalg.norm(G(p))) for p in P)
check("toy is (L0, L1)-smooth with L0 = 1, L1 = 3 on [-6, 6]^2", ratio <= 1, f"max |H|/(1 + 3|g|) = {ratio:.3f}")
check("toy is not L-smooth: curvature 1 + 3x^2 is unbounded", 1 + 3 * 100 ** 2 > 1e4)
EPS = 1e-4


def iters(step, z0, cap=200000):
    z = np.array(z0, float)
    for t in range(cap):
        g = G(z)
        if np.linalg.norm(g) <= EPS: return t
        z = step(z, g)
        if not np.all(np.isfinite(z)) or np.abs(z).max() > 1e6: return None
    return None


def gd(h):
    return lambda z, g: z - h * g


def clip(eta, gam):
    return lambda z, g: z - min(eta, gam * eta / np.linalg.norm(g)) * g


X0S = [1, 2, 3, 4, 6, 8]
HS = np.geomspace(1e-4, 1.9, 60)
rows = []
for x0 in X0S:
    z0 = [x0, 1.0]
    best = min((n, h) for h in HS if (n := iters(gd(h), z0)) is not None)
    nc = iters(clip(1.0, .5), z0)
    M = np.linalg.norm(G(z0))
    rows.append([x0, round(float(M), 2), best[0], round(float(best[1]), 5), nc])
    print(f"    x0 = {x0}: M = {M:.1f}; best fixed-step GD {best[0]} iters (h = {best[1]:.4f}); clipped GD (eta 1, gamma 0.5) {nc} iters")
check("fixed-step GD's best iteration count grows with the initial gradient size M", rows[-1][2] > 10 * rows[0][2])
check("clipped GD with one setting stays fast for every start", max(r_[4] for r_ in rows) < 60, f"{[r_[4] for r_ in rows]}")
check("clipping is >10x faster than the best fixed step from x0 = 8", rows[-1][2] > 10 * rows[-1][4], f"{rows[-1][2]} vs {rows[-1][4]}")
save_json("zhang2020gradient", "data.json", {"rows": rows, "eps": EPS})
done()
