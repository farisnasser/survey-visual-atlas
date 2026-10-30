"""Taniguchi et al. 2024: ADOPT, modified Adam that converges with any beta2 at the optimal rate.

Algorithm 1 (ADOPT): v0 = g0^2; m1 = g1 / max(sqrt(v0), eps); for t >= 1: theta_t = theta_{t-1} - eta_t m_t;
v_t = b2 v_{t-1} + (1 - b2) g_t^2; m_{t+1} = b1 m_t + (1 - b1) g_{t+1} / max(sqrt(v_t), eps).
(normalize the fresh gradient by the *previous* second moment, then apply momentum.)
Toy (Sec. 5, Eq. 17): f(theta) = theta on [-1, 1]; stochastic gradient k^2 w.p. 1/k, -k otherwise (unbiased, E g = 1);
eta_t = 0.01 / sqrt(1 + 0.01 t), b1 = 0.9. Solution theta* = -1. Adam fails for small b2 (and for k = 50 even at 0.999).
"""
from common import *

print("taniguchi2024adopt")
B1, EPS = .9, 1e-8


def grads(k, T, seed):
    r = np.random.default_rng(seed)
    return np.where(r.random(T) < 1 / k, k * k, -k).astype(float)


def run_all(k, T, seeds, b2s):
    """all three methods at once, vectorized over (beta2, seed); returns final theta arrays and checkpoints of the mean."""
    r = np.random.default_rng(seeds)
    S, nb = 8, len(b2s)
    b2 = np.array(b2s)[:, None] * np.ones((1, S))
    G_ = lambda: np.where(r.random(S) < 1 / k, k * k, -k).astype(float)[None, :] * np.ones((nb, 1))
    tha = np.zeros((nb, S)); ma = np.zeros((nb, S)); va = np.zeros((nb, S))
    ths = np.zeros((nb, S)); ms_ = np.zeros((nb, S)); vs = np.zeros((nb, S)); vmax = np.zeros((nb, S))
    g0 = G_(); g1 = G_()
    tho = np.zeros((nb, S)); vo = g0 ** 2; mo = g1 / np.maximum(np.sqrt(vo), EPS); gnext = g1
    marks = {}
    for t in range(1, T + 1):
        eta = .01 / np.sqrt(1 + .01 * t)
        g = G_()
        # Adam / AMSGrad (bias-corrected), projected on [-1, 1]
        ma = B1 * ma + (1 - B1) * g; va = b2 * va + (1 - b2) * g * g
        tha = np.clip(tha - eta * (ma / (1 - B1 ** t)) / (np.sqrt(va / (1 - b2 ** t)) + EPS), -1, 1)
        ms_ = B1 * ms_ + (1 - B1) * g; vs = b2 * vs + (1 - b2) * g * g; vmax = np.maximum(vmax, vs / (1 - b2 ** t))
        ths = np.clip(ths - eta * (ms_ / (1 - B1 ** t)) / (np.sqrt(vmax) + EPS), -1, 1)
        # ADOPT: theta_t uses m_t (built from g_t normalized by v_{t-1}); then v_t; then m_{t+1}
        tho = np.clip(tho - eta * mo, -1, 1)
        vo = b2 * vo + (1 - b2) * gnext ** 2
        gnext = g
        mo = B1 * mo + (1 - B1) * gnext / np.maximum(np.sqrt(vo), EPS)
        if t in CHECKS:
            marks[t] = (tha.mean(1).round(3).tolist(), ths.mean(1).round(3).tolist(), tho.mean(1).round(3).tolist())
    return tha, ths, tho, marks


check("toy gradient is unbiased: E g = k^2/k - k(1 - 1/k) = 1", all(abs(k * k / k - k * (1 - 1 / k) - 1) < 1e-12 for k in [10, 50]))
g = grads(10, 400000, 0)
check("Monte Carlo mean of the stochastic gradient ~ 1 (k = 10)", abs(g.mean() - 1) < .1, f"{g.mean():.3f}")
T = 400000
CHECKS = {1000, 10000, 100000, 400000}
B2S = [.1, .5, .9, .99, .999]
res, traj = {}, {}
for k in [10, 50]:
    a, s_, o, marks = run_all(k, T, k, B2S)
    traj[str(k)] = {str(t): v for t, v in marks.items()}
    for i, b2 in enumerate(B2S):
        res[f"{k}-{b2}"] = {"adam": round(float(a[i].mean()), 3), "amsgrad": round(float(s_[i].mean()), 3), "adopt": round(float(o[i].mean()), 3)}
        print(f"    k={k} b2={b2}: " + ", ".join(f"{m} {v:+.3f}" for m, v in res[f'{k}-{b2}'].items()))
check("k = 10: ADOPT reaches theta* = -1 for every beta2 (8 seeds, 4e5 steps)", all(res[f"10-{b2}"]["adopt"] < -.99 for b2 in B2S))
ad50 = [traj["50"][str(t)][2] for t in sorted(CHECKS)[1:]]
check("k = 50: ADOPT moves steadily towards -1 for every beta2 (not yet there at 4e5 steps)", all(ad50[i][j] >= ad50[i + 1][j] for i in range(len(ad50) - 1) for j in range(len(B2S))) and all(v < -.2 for v in ad50[-1]), f"theta at 1e4/1e5/4e5 steps: {ad50}")
check("Adam fails (theta > 0) for small beta2 with k = 10", all(res[f"10-{b2}"]["adam"] > 0 for b2 in [.1, .5, .9]))
check("Adam with k = 50 fails even at beta2 = 0.999 (paper Fig. 1)", res["50-0.999"]["adam"] > -.5, f"{res['50-0.999']['adam']}")
check("Adam with k = 10 succeeds at beta2 = 0.999 (paper Fig. 1)", res["10-0.999"]["adam"] < -.9, f"{res['10-0.999']['adam']}")
check("AMSGrad is slower than ADOPT at k = 50", all(res[f"50-{b2}"]["amsgrad"] > res[f"50-{b2}"]["adopt"] for b2 in B2S))
# the notes' ordering example
check("notes: v = 0.9*1 + 0.1*16 = 2.5, 4/sqrt(2.5) = 2.53 vs 4/sqrt(1) = 4", abs(.9 + .1 * 16 - 2.5) < 1e-12 and abs(4 / np.sqrt(2.5) - 2.5298) < 1e-4)
save_json("taniguchi2024adopt", "data.json", {"T": T, "b2s": B2S, "res": res, "traj": traj})
done()
