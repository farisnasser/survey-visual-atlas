"""Sun 2020 ("Optimization for deep learning: an overview", J. Oper. Res. Soc. China 8; arXiv 1912.08957), Sec. 4.1:
gradient explosion / vanishing. Scalar deep linear net, F(w) = (w_1 ... w_L - 1)^2 (Eq. 13): with all w_j = 2 the gradient has norm
~2^(L-1)|e|, with all w_j = 1/2 ~0.5^(L-1)|e| (e = prod w - 1). Example: F(w) = (w^7 - 1)^2, flat on [-1+c, 1-c] (vanishing),
steep outside [-1-c, 1+c] (exploding) (Fig. 2).
Toy on the page: slice F(a, b) = (a b^(L-1) - 1)^2 of the depth-L chain (w_1 = a, w_2 = ... = w_L = b).
"""
from common import *

print("sun2020optimization")
for L in [2, 4, 7]:
    Fs = lambda p, L=L: (p[0] * p[1] ** (L - 1) - 1) ** 2
    gs = lambda p, L=L: np.array([2 * (p[0] * p[1] ** (L - 1) - 1) * p[1] ** (L - 1), 2 * (p[0] * p[1] ** (L - 1) - 1) * (L - 1) * p[0] * p[1] ** (L - 2)])
    check_grad(f"slice, L = {L}", Fs, gs, rand_pts(20, -1.3, 1.3))
# the paper's statement for the chain F(w) = (prod w_j - 1)^2: dF/dw_k = 2 e prod_{j != k} w_j
for L in [5, 10, 20]:
    for s in [.5, 2.]:
        w = np.full(L, s); e = np.prod(w) - 1; gk = 2 * e * np.prod(w[1:])
        Fw = lambda v: (np.prod(v) - 1) ** 2
        fd = fd_grad(Fw, w, h=1e-7 * max(1, s ** L))[0]
        check(f"L = {L}, all w = {s}: dF/dw_1 = 2 e s^(L-1) = {gk:.3g}", abs(fd - gk) < 1e-4 * max(1, abs(gk)))
# Fig. 2: F(w) = (w^7 - 1)^2: |F'| on [-0.8, 0.8] vs on [1.2, 1.3]
F7 = lambda w: (w ** 7 - 1) ** 2; d7 = lambda w: 14 * w ** 6 * (w ** 7 - 1)
xi = np.linspace(-.8, .8, 1601); xo = np.linspace(1.2, 1.3, 101)
print(f"  F = (w^7 - 1)^2: max |F'| on [-0.8, 0.8] = {np.abs(d7(xi)).max():.3f}; min |F'| on [1.2, 1.3] = {np.abs(d7(xo)).min():.1f}")
xc = np.linspace(-.4, .4, 801)
check("flat: |F'| <= 4.5 on [-0.8, 0.8] and < 0.1 on [-0.4, 0.4], while F stays above 0.6 (minimum 0 at w = 1)", np.abs(d7(xi)).max() < 4.5 and np.abs(d7(xc)).max() < .1 and F7(xi).min() > .6)
check("steep: |F'| > 100 on [1.2, 1.3], more than 20x the flat-region maximum", np.abs(d7(xo)).min() > 20 * np.abs(d7(xi)).max())
# GD on (w^7 - 1)^2: from w0 = 0.5 almost no progress for many steps; from w0 = 1.25 a fixed step that is fine near 1 overshoots
def gd1(w, eta, T):
    out = [w]
    for _ in range(T):
        w = w - eta * d7(w); out.append(w)
        if not np.isfinite(w) or abs(w) > 1e3: break
    return np.array(out)
ETA = .02                                   # F''(1) = 98, so 0.02 < 2/98 is a stable step at the minimum
a = gd1(.3, ETA, 3000); b = gd1(1.3, ETA, 50); c = gd1(1.02, ETA, 3000)
k99 = int(np.argmax(np.abs(a - 1) < .01)) if np.any(np.abs(a - 1) < .01) else -1
print(f"  GD step 0.02: from 0.3 reaches |w - 1| < 0.01 after {k99} steps; from 1.3: |w| = {np.abs(b[:4]).round(3).tolist()} ... (diverges: {len(b) < 51}); from 1.02 converges to {c[-1]:.6f}")
check("vanishing: from w0 = 0.3 GD needs > 300 steps", k99 > 300)
check("exploding: from w0 = 1.3 the same step diverges", len(b) < 51)
check("and the step is fine near the minimum: from 1.02 GD converges", abs(c[-1] - 1) < 1e-9)
# the page slice F(a, b) = (a b^(L-1) - 1)^2, GD step 0.02, from (0.3, 0.3), (0.9, 0.9), (1.25, 1.25); box |a|, |b| <= 1.4
slres = {}
for L in [2, 4, 7]:
    row = []
    for st in [(.3, .3), (.9, .9), (1.25, 1.25)]:
        p_ = np.array(st); k = 0; res_ = None
        for k in range(3000):
            e = p_[0] * p_[1] ** (L - 1) - 1
            if e * e < 1e-6: res_ = k; break
            p_ = p_ - ETA * np.array([2 * e * p_[1] ** (L - 1), 2 * e * (L - 1) * p_[0] * p_[1] ** (L - 2)])
            if np.max(np.abs(p_)) > 1.4: res_ = "out"; break
        row.append(res_)
    slres[L] = row
    print(f"  slice L = {L}: steps to loss < 1e-6 from (0.3,0.3), (0.9,0.9), (1.25,1.25): {row}")
check("L = 7: the start on the cliff leaves the box; the plateau start needs > 3x the steps it needs at L = 2", slres[7][2] == "out" and slres[7][0] > 3 * slres[2][0])
check("L = 2: all three starts converge", all(isinstance(v, int) for v in slres[2]))
save_json("sun2020optimization", "data.json", {"traj": {"0.3": r(a[:3001:10], 6), "1.3": [float(f"{q:.4g}") for q in b[:6]]}, "k99": k99})
done()
