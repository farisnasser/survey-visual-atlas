"""He, Zhang, Ren, Sun 2016: deep residual learning.

Residual block (Eq. 1): y = F(x, {W_i}) + x (identity shortcut); the layers fit F(x) = H(x) - x.
Toy for the 3D tab: a scalar deep linear chain of depth 2K with tied weights, a on odd and b on even layers, input x = 1,
target 2, loss 1/2 (f - 2)^2.
  plain:    f = a^K b^K
  residual: f = (1 + a)^K (1 + b)^K      (each layer computes h -> h + w h)
At a small initialization the plain chain's gradient vanishes like K a^{K-1} b^K, the residual chain's does not.
Reported (Table 2, ImageNet top-1 %, 10-crop): plain-18 27.94, plain-34 28.54; ResNet-18 27.88, ResNet-34 25.03.
"""
from common import *

print("he2016deep")
Y = 2.0


def mk(K, res):
    if res:
        f = lambda z: (1 + z[0]) ** K * (1 + z[1]) ** K
        g = lambda z: (f(z) - Y) * np.array([K * (1 + z[0]) ** (K - 1) * (1 + z[1]) ** K, K * (1 + z[0]) ** K * (1 + z[1]) ** (K - 1)])
    else:
        f = lambda z: z[0] ** K * z[1] ** K
        g = lambda z: (f(z) - Y) * np.array([K * z[0] ** (K - 1) * z[1] ** K, K * z[0] ** K * z[1] ** (K - 1)])
    return (lambda z: .5 * (f(z) - Y) ** 2), g


for K in [1, 3, 10]:
    for res in [False, True]:
        L, g = mk(K, res)
        check_grad(f"K = {K} {'residual' if res else 'plain'}", L, g, rand_pts(4, .05, 1.1))
Z0 = np.array([.02, .03])


def run(K, res, eta, T=2000):
    L, g = mk(K, res); z = Z0.copy(); out = []
    for _ in range(T):
        z = z - eta * g(z); out.append(L(z))
        if not np.all(np.isfinite(z)) or np.abs(z).max() > 10: break
    return np.array(out), z


rows = []
for K in [1, 3, 10]:
    Lp, gp = mk(K, False); Lr, gr = mk(K, True)
    gpn, grn = np.linalg.norm(gp(Z0)), np.linalg.norm(gr(Z0))
    best_p = min(((run(K, False, e)[0][-1], e) for e in [.01, .03, .1, .3, 1, 3, 10]), key=lambda t: t[0] if np.isfinite(t[0]) else np.inf)
    best_r = min(((run(K, True, e)[0][-1], e) for e in [.001, .003, .01, .03, .1, .3]), key=lambda t: t[0] if np.isfinite(t[0]) else np.inf)
    rows.append([K, float(f"{gpn:.3g}"), float(f"{grn:.3g}"), float(f"{best_p[0]:.3g}"), best_p[1], float(f"{best_r[0]:.3g}"), best_r[1]])
    print(f"    depth {2 * K}: |grad| at init plain {gpn:.2e} vs residual {grn:.2e}; loss after 2000 GD steps (best eta): plain {best_p[0]:.2e} (eta {best_p[1]}), residual {best_r[0]:.2e} (eta {best_r[1]})")
check("depth 20: plain gradient at init is vanishingly small (< 1e-15)", rows[-1][1] < 1e-15)
check("residual gradient at init stays order one for every depth", all(r_[2] > .5 for r_ in rows))
check("depth 20: plain chain makes no progress (loss stays 2.0), residual reaches ~0", rows[-1][3] > 1.99 and rows[-1][5] < 1e-10)
check("depth 2: both train (the shallow plain net is fine)", rows[0][3] < 1e-6 and rows[0][5] < 1e-6)
save_json("he2016deep", "data.json", {"rows": rows, "table2": {"plain": [27.94, 28.54], "resnet": [27.88, 25.03], "depth": [18, 34]}})
done()
