"""Antonakopoulos, Mertikopoulos, Piliouras, Wang 2022 (ICML, PMLR 162): deterministic AdaGrad x_{t+1} = x_t - Gamma_t grad f(x_t)
(ADANORM)  Gamma_t = 1 / sqrt(d0^2 + sum_{s<=t} |g_s|^2)
(ADADIAG)  Gamma_t = G_t^{-1/2},  G_t = d0^2 I + diag(sum_{s<=t} g_s g_s^T)
(ADAFULL)  Gamma_t = G_t^{-1/2},  G_t = d0^2 I + sum_{s<=t} g_s g_s^T
Proposition 1: Gamma_t -> Gamma > 0 and sum |grad f(x_t)|^2 < inf.  Theorem 2: all three avoid strict saddles from almost every start.
Proposition 3 asks d0 >= L.

Toy: f = 5 u^2 - v^2/2 + v^4/4 in coordinates rotated by 30 deg (u = c x + s y, v = -s x + c y): strict saddle at 0 with Hessian
eigenvalues (10, -1) not aligned with the axes (so the diagonal preconditioner does not match the curvature), minima v = +-1, u = 0.
"""
from common import *

print("antonakopoulos2022adagrad")
th = np.pi / 6; c, s = np.cos(th), np.sin(th)
uv = lambda p: (c * p[..., 0] + s * p[..., 1], -s * p[..., 0] + c * p[..., 1])
def f(p): u, v = uv(np.asarray(p, float)); return 5 * u ** 2 - v ** 2 / 2 + v ** 4 / 4
def g(p):
    p = np.asarray(p, float); u, v = uv(p); gu, gv = 10 * u, -v + v ** 3
    return np.stack([c * gu - s * gv, s * gu + c * gv], -1)
check_grad("rotated f", f, g, rand_pts(20, -2, 2))
H0 = fd_hess(f, [0, 0]); ev = eig(H0)
check("saddle at 0: Hessian eigenvalues (-1, 10), eigenvectors at 30 deg to the axes", abs(ev[0] + 1) < 1e-3 and abs(ev[1] - 10) < 1e-3 and abs(H0[0, 1]) > 1)
mins = np.array([[-s, c], [s, -c]])
check("minima at v = +-1, u = 0, f = -1/4", all(np.linalg.norm(g(m)) < 1e-12 and abs(f(m) + .25) < 1e-12 for m in mins))
D0 = 10.
def ada(P, kind, T=3000):
    P = np.array(P, float); acc = np.zeros(P.shape[:-1] + (2, 2)); S = 0; gsum = np.zeros(P.shape[:-1])
    for t in range(T):
        G = g(P); gsum = gsum + np.sum(G ** 2, -1)
        if kind == "gd": step = .09 * G
        elif kind == "norm": step = G / np.sqrt(D0 ** 2 + gsum)[..., None]
        elif kind == "diag":
            acc[..., 0, 0] += G[..., 0] ** 2; acc[..., 1, 1] += G[..., 1] ** 2
            step = G / np.sqrt(D0 ** 2 + np.stack([acc[..., 0, 0], acc[..., 1, 1]], -1))
        else:
            acc += G[..., :, None] * G[..., None, :]; M = D0 ** 2 * np.eye(2) + acc
            w, V = np.linalg.eigh(M); Ginv = (V * (1 / np.sqrt(w))[..., None, :]) @ np.swapaxes(V, -1, -2)
            step = np.einsum("...ij,...j->...i", Ginv, G)
        P = P - step
    return P, gsum
def gamma_hist(p0, kind, T=3000):
    p = np.array(p0, float); acc = np.zeros((2, 2)); gs = 0; out = []
    for t in range(T):
        G = g(p); gs += G @ G
        if kind == "diag": acc += np.diag(G ** 2); Gm = np.diag(1 / np.sqrt(D0 ** 2 + np.diag(acc)))
        elif kind == "full": acc += np.outer(G, G); w, V = np.linalg.eigh(D0 ** 2 * np.eye(2) + acc); Gm = V @ np.diag(1 / np.sqrt(w)) @ V.T
        else: Gm = np.eye(2) / np.sqrt(D0 ** 2 + gs)
        out.append(np.linalg.eigvalsh(Gm)); p = p - Gm @ G
    return np.array(out), p, gs
start = list(1.5 * np.array([c, s]) + 1e-3 * np.array([-s, c]))   # 1e-3 off the saddle's stable manifold (the u-axis)
dist_sm = abs(uv(np.array(start))[1]); print(f"  start {start}: distance to the saddle's stable manifold v = 0: {dist_sm:.4f}")
fin = {}
for kind in ["gd", "norm", "diag", "full"]:
    P, gs = ada(np.array([start]), kind, 3000); fin[kind] = P[0]
    print(f"  {kind:5s}: ends at ({P[0, 0]:.4f}, {P[0, 1]:.4f}), f = {f(P[0]):.5f}, sum |grad|^2 = {gs[0]:.3f}")
check("GD and all three AdaGrad variants reach a minimum from the near-saddle start", all(min(np.linalg.norm(fin[k] - m) for m in mins) < 1e-3 for k in fin))
for kind in ["norm", "diag", "full"]:
    lam, p, gs = gamma_hist(start, kind, 6000)
    check(f"Prop. 1 ({kind}): eigenvalues of Gamma_t settle at positive limits (change < 1e-6 over the last 1000 steps)", np.all(lam[-1] > 0) and np.max(np.abs(lam[-1] - lam[-1000])) < 1e-6, f"limit {np.round(lam[-1], 4)}")
# escape time vs offset d from each method's own stable set. Along the segment p(d) = 1.5 (c, s) + d (-s, c), the stable set of the
# saddle is where the final minimum flips; find that d* by bisection (for GD it is d* = 0, the Hessian's stable manifold v = 0).
def step_fn(kind):
    def run(P, T, track=False):
        P = np.array(P, float); acc = np.zeros((len(P), 2, 2)); gs = np.zeros(len(P)); tt = np.full(len(P), -1)
        for t in range(T):
            G = g(P); gs += np.sum(G ** 2, -1)
            if kind == "gd": st = .09 * G
            elif kind == "norm": st = G / np.sqrt(D0 ** 2 + gs)[:, None]
            elif kind == "diag": acc[:, 0, 0] += G[:, 0] ** 2; acc[:, 1, 1] += G[:, 1] ** 2; st = G / np.sqrt(D0 ** 2 + np.stack([acc[:, 0, 0], acc[:, 1, 1]], -1))
            else:
                acc += G[:, :, None] * G[:, None, :]; w, V = np.linalg.eigh(D0 ** 2 * np.eye(2) + acc)
                st = np.einsum("nij,nj->ni", (V * (1 / np.sqrt(w))[:, None, :]) @ np.swapaxes(V, 1, 2), G)
            P = P - st; tt = np.where((tt < 0) & (np.abs(uv(P)[1]) > .5), t + 1, tt)
        return P, tt
    return run
base, nrm = 1.5 * np.array([c, s]), np.array([-s, c])
side = lambda P: np.sign(uv(P)[1])
dstar, esc = {}, {}
ds = np.logspace(-12, -1, 12)
for kind in ["gd", "norm", "diag", "full"]:
    run = step_fn(kind); lo, hi = -.5, .5
    slo = side(run(base[None] + lo * nrm[None], 3000)[0])[0]
    for _ in range(60):
        mid = (lo + hi) / 2; sm = side(run(base[None] + mid * nrm[None], 3000)[0])[0]
        if sm == slo: lo = mid
        else: hi = mid
    dstar[kind] = (lo + hi) / 2
    _, tt = run(base[None] + (dstar[kind] + ds)[:, None] * nrm[None], 6000)
    esc[kind] = tt.tolist(); sl = np.polyfit(np.log(1 / ds), tt, 1)[0]
    print(f"  {kind:5s}: stable set crosses the segment at d* = {dstar[kind]:+.6f}; escape steps from d* + 1e-12 ... d* + 1e-1: {tt.tolist()}; slope per unit log(1/d) {sl:.2f}")
check("GD: the stable set is the Hessian's stable manifold (d* = 0)", abs(dstar["gd"]) < 1e-9)
check("AdaGrad-diag: its stable set is elsewhere (|d*| > 0.01): the preconditioner is not aligned with the curvature", abs(dstar["diag"]) > .01)
check("every start off a method's stable set escapes, in time growing like log(1/d) (no uniform bound)", all(min(v) > 0 for v in esc.values()) and all(np.polyfit(np.log(1 / ds), esc[k], 1)[0] > 3 for k in esc))
save_json("antonakopoulos2022adagrad", "data.json", {"ds": ds.tolist(), "esc": esc, "dstar": {k: float(v) for k, v in dstar.items()}, "d0": D0})
done()
