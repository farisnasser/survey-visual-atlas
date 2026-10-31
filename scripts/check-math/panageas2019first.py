"""Panageas, Piliouras, Wang 2019 (NeurIPS; arXiv 1906.07772v3): GD (and mirror descent, proximal point, manifold GD) with
vanishing steps alpha_k = Omega(1/k) avoids strict saddles for almost every start (Theorems 4.1, 5.1-5.6; v3 adds the
assumption that the Hessian is invertible). Remark 3.1 / Fig. 1: summable steps (1/k^2, 1/k^4) can converge to a NON-critical point.

Toy: f(x, y) = x^2/2 - y^2/2 + y^4/4: strict saddle at 0 with invertible Hessian diag(1, -1), minima (0, +-1).
Schedules alpha_k = c / (k+1)^s with c = 0.8 and s = 1/2, 1, 2.
"""
from common import *

print("panageas2019first")
f = lambda p: p[0] ** 2 / 2 - p[1] ** 2 / 2 + p[1] ** 4 / 4
g = lambda p: np.array([p[0], -p[1] + p[1] ** 3])
check_grad("f", f, g, rand_pts(20, -2, 2))
check_hess_at("at 0 = diag(1, -1): strict saddle, invertible", f, [0, 0], [[1, 0], [0, -1]])
check_hess_at("at (0, 1) = diag(1, 2): minimum", f, [0, 1], [[1, 0], [0, 2]])
C, K = .8, 4000
def run(P, s, K=K):
    P = np.array(P, float)
    for k in range(K):
        a = C / (k + 1) ** s; P = P - a * np.stack([P[..., 0], -P[..., 1] + P[..., 1] ** 3], -1)
    return P
start = [1.5, .05]
ends = {s: run(start, s) for s in (.5, 1, 2)}
for s, p in ends.items():
    print(f"  alpha_k = 0.8/(k+1)^{s}: after {K} steps p = ({p[0]:.4g}, {p[1]:.4g}), |grad f| = {np.linalg.norm(g(p)):.3g}")
check("s = 1/2 and s = 1 (not summable, Omega(1/k)): reach the minimum (0, 1)", all(np.linalg.norm(ends[s] - [0, 1]) < 1e-3 for s in (.5, 1)))
S2 = C * np.pi ** 2 / 6
xlim = 1.5 * np.prod([1 - C / (k + 1) ** 2 for k in range(200000)])
check("s = 2 (summable, sum = 0.8 pi^2/6 = 1.316): converges to a NON-critical point, x -> 1.5 prod(1 - 0.8/(k+1)^2)", np.linalg.norm(g(ends[2])) > .1 and abs(ends[2][0] - xlim) < 1e-3, f"x_inf = {xlim:.4f}")
# linearized escape along the unstable direction: y_k ~ y_0 prod(1 + alpha_j); for s = 1 this grows like k^0.8
yl = .05 * np.prod([1 + C / (j + 1) for j in range(400)])
print(f"  s = 1, linearized y after 400 steps: {yl:.3f} (~ 0.05 * 400^0.8 / Gamma(1.8) = {.05 * 401 ** .8 / 0.9314:.3f})")
# stable set of the saddle: grid of starts, classify the endpoints
xs = np.linspace(-2, 2, 121); ys = np.linspace(-1.6, 1.6, 97); XX, YY = np.meshgrid(xs, ys); G0 = np.stack([XX, YY], -1)
frac = {}
for s in (.5, 1, 2):
    E = run(G0, s, 3000)
    toMin = np.minimum(np.linalg.norm(E - [0, 1], axis=-1), np.linalg.norm(E + [0, 1], axis=-1)) < .05
    toSad = np.linalg.norm(E, axis=-1) < .05
    other = ~toMin & ~toSad
    frac[s] = (float(toMin.mean()), float(toSad.mean()), float(other.mean()), int(toSad.sum()))
    rows_ = sorted(set(np.round(YY[toSad], 4).tolist()))
    if s < 2: print(f"    starts reaching the saddle lie on the rows y = {rows_}")
    if s < 2: SADROWS = rows_
    print(f"  grid 121 x 97, alpha = 0.8/(k+1)^{s}, 3000 steps: at a minimum {toMin.mean():.4f}, at the saddle {toSad.mean():.4f} ({toSad.sum()} starts), elsewhere {other.mean():.4f}")
check("Omega(1/k) steps: starts that reach the saddle lie on y = 0 or on y = +-1.5, which the first step (alpha_0 = 0.8) maps exactly onto y = 0 (both measure zero)", SADROWS == [-1.5, 0.0, 1.5] and abs(1.5 * (1 + .8 - .8 * 1.5 ** 2)) < 1e-12)
check("Omega(1/k) steps: every other start reaches a minimum", frac[.5][2] == 0 and frac[1][2] == 0)
check("summable steps: a whole open band of starts converges to non-critical points (positive measure)", frac[2][2] > .3)
save_json("panageas2019first", "data.json", {"frac": {str(s): v for s, v in frac.items()}, "xinf": round(float(xlim), 4), "ends": {str(s): r(p, 5) for s, p in ends.items()}})

# in-page notes (papers/panageas2019first/notes.md): Sec. 2/4 linearization and the primer's numbers
gam, ak = .7, [.5 / (k + 1) for k in range(200)]
z = 1e-3
for a in ak:
    z = z * (1 + gam * a)
check("notes Sec. 2: z_t = z_0 prod(1 + gamma alpha_k) on the unstable coordinate", np.isclose(z, 1e-3 * np.prod([1 + gam * a for a in ak])))
t_ = 1000
check("notes Sec. 4: log prod(1 + g a_k) ~ g sum a_k for small steps (ratio -> 1)",
      abs(np.sum(np.log1p(gam * .01 / np.arange(1, 10 ** 5))) / (gam * np.sum(.01 / np.arange(1, 10 ** 5))) - 1) < 1e-2)
check("primer idea 1: prod_{k=1}^t (1 + 1/k) = t + 1; prod (1 + 1/k^2) -> sinh(pi)/pi ~ 3.68",
      np.isclose(np.prod(1 + 1 / np.arange(1, t_ + 1)), t_ + 1) and abs(np.prod(1 + 1 / np.arange(1, 10 ** 6) ** 2) - np.sinh(np.pi) / np.pi) < 1e-5 and round(np.sinh(np.pi) / np.pi, 2) == 3.68)
xinf = 1.5 * np.prod(1 - .8 / np.arange(1, 10 ** 6) ** 2)
check("primer idea 2: x_inf = 1.5 prod(1 - 0.8/(k+1)^2) ~ 0.174 (not 0)", round(xinf, 3) == .174, f"{xinf:.4f}")
grow = lambda T: .05 * np.prod(1 + .8 / np.arange(1, T + 1))
check("primer idea 3: with alpha_k = 0.8/(k+1) the unstable coordinate keeps growing (~ t^0.8)",
      grow(5000) / grow(500) > 6 and grow(50000) / grow(5000) > 6, f"{grow(500):.1f} -> {grow(5000):.1f} -> {grow(50000):.1f}")
xm = np.array([.5, .5]) * np.exp(-1 * np.array([1., 0])); xm /= xm.sum()
check("primer idea 4: entropic mirror step from (0.5,0.5), grad (1,0), alpha 1 -> (0.269, 0.731)", np.allclose(np.round(xm, 3), [.269, .731]))
done()
