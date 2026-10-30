"""Chen, Zhou, Gu 2022: LENA (Last stEp shriNkAge, Algorithm 1) with the SPIDER estimator (Algorithm 2), online setting.

F(x, y) = (x^2 - 1)^2/4 + y^2/2;  stochastic gradient grad f(x; xi) = grad F + (c x, b - c y), xi = (b, c) ~ N(0, diag(sa^2, sc^2)).
The x-noise vanishes on the line x = 0 (the stable manifold of the saddle), so only the perturbation can start the escape.
GD phase: while |d_t| > eps: x <- x - (eta/|d_t|) d_t (normalized).  Then perturb x += xi ~ Unif(B(r)) and run up to l_thres escape
steps with eta_H; if sum_{i=m_s}^t eta_i^2 |d_i|^2 > (t - m_s + 1) D, shrink the last step so equality holds, take it and break
(escape happened -> new GD phase).  If l_thres steps pass without that, output x_{m_s}.
SPIDER (Alg. 2): every q steps a big batch of B samples, otherwise d <- d + mean_b [grad f(x_{t+1}; xi) - grad f(x_t; xi)] (same xi).
"""
from common import *

print("chen2022faster")
F = lambda p: .25 * (p[0] ** 2 - 1) ** 2 + .5 * p[1] ** 2
G = lambda p: np.array([p[0] ** 3 - p[0], p[1]])
sa, sc = .5, .3
gs = lambda p, X: np.stack([p[0] ** 3 - p[0] + X[:, 2] * p[0], p[1] + X[:, 1] - X[:, 2] * p[1]], -1)
check_grad("F", F, G, rand_pts(20, -1.6, 1.6))
Xbig = np.random.default_rng(0).normal(size=(200000, 3)) * [sa, sa, sc]
check("stochastic gradient is unbiased", np.allclose(gs(np.array([.3, .4]), Xbig).mean(0), G([.3, .4]), atol=5e-3))
B, eps = 400, .05
b = q = int(np.sqrt(B))
eps_H = np.sqrt(eps)
eta, etaH, r_, D, lthres = .03, .1, .01, 1e-4, 120   # l_thres = O~(1/(eta_H eps_H)): 1/(0.1*0.224) = 45 times a log factor
print(f"  B = {B}, b = q = {b}, eps = {eps}, eps_H = sqrt(eps) = {eps_H:.3f}, eta = {eta}, eta_H = {etaH}, D = {D}, l_thres = {lthres}")


def lena(seed, x0=(0., 1.3), Tmax=6000):
    R = np.random.default_rng(seed)
    x = np.array(x0); t = 0; sfo = B
    d = gs(x, R.normal(size=(B, 3)) * [sa, sa, sc]).mean(0)
    log = {"shrinks": 0, "phases": 0, "err": []}

    def est(x_old, x_new, d):
        nonlocal t, sfo
        t += 1
        if t % q == 0:
            sfo += B; return gs(x_new, R.normal(size=(B, 3)) * [sa, sa, sc]).mean(0)
        X = R.normal(size=(b, 3)) * [sa, sa, sc]; sfo += 2 * b
        return d + (gs(x_new, X) - gs(x_old, X)).mean(0)

    while t < Tmax:
        log["phases"] += 1
        while np.linalg.norm(d) > eps and t < Tmax:
            xn = x - eta / np.linalg.norm(d) * d; d = est(x, xn, d); x = xn; log["err"].append(np.linalg.norm(d - G(x)))
        ms = x.copy()
        th = R.uniform(0, 2 * np.pi); rad = r_ * np.sqrt(R.uniform())
        xn = x + rad * np.array([np.cos(th), np.sin(th)]); d = est(x, xn, d); x = xn
        mov, found = 0.0, True
        for k in range(lthres):
            step2 = etaH ** 2 * (d @ d)
            if mov + step2 > (k + 1) * D:
                eta_t = np.sqrt(max((k + 1) * D - mov, 0) / (d @ d))   # last step shrinkage: equality
                xn = x - eta_t * d; d = est(x, xn, d); x = xn; log["shrinks"] += 1; found = False; break
            mov += step2; xn = x - etaH * d; d = est(x, xn, d); x = xn; log["err"].append(np.linalg.norm(d - G(x)))
        if found:
            return ms, t, sfo, log
    return ms, t, sfo, log


runs = [lena(s) for s in range(30)]
ok = [np.linalg.norm(G(o[0])) <= 2 * eps and eig(np.diag([3 * o[0][0] ** 2 - 1, 1]))[0] >= -eps_H for o in runs]
check("LENA outputs a (2 eps, eps_H)-approximate local minimum in every run (estimator error allowed)", all(ok),
      f"output |grad F| median {np.median([np.linalg.norm(G(o[0])) for o in runs]):.3f}, x median |x1| {np.median([abs(o[0][0]) for o in runs]):.3f}")
check("at least one last-step shrinkage (escape detected) per run", min(o[3]["shrinks"] for o in runs) >= 1, f"median shrinks {np.median([o[3]['shrinks'] for o in runs]):.0f}")
sfo_med = float(np.median([o[2] for o in runs]))
print(f"  stochastic gradients used: median {sfo_med:.0f}")
errs = np.concatenate([o[3]["err"] for o in runs])
check("SPIDER estimator error stays O(eps) along the run (Eq. 4.1 keeps it small)", np.percentile(errs, 95) < 2 * eps, f"95th pct {np.percentile(errs, 95):.4f}")
check("notes' shrink example: sqrt(0.003 - 0.03^2 - 0.04^2) = 0.0224", abs(np.sqrt(.003 - .03 ** 2 - .04 ** 2) - .02236) < 1e-4)
save_json("chen2022faster", "data.json", {"sa": sa, "sc": sc, "B": B, "b": b, "eps": eps, "epsH": round(float(eps_H), 5), "eta": eta, "etaH": etaH, "r": r_, "D": D, "lthres": lthres,
                                         "sfo_med": sfo_med})
done()
