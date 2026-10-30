"""Kingma & Ba 2015: Adam.

Algorithm 1: m_t = b1 m + (1 - b1) g; v_t = b2 v + (1 - b2) g^2; m_hat = m/(1 - b1^t); v_hat = v/(1 - b2^t);
theta <- theta - alpha m_hat / (sqrt(v_hat) + eps).
Sec. 2.1: effective step |Delta_t| <= alpha (1 - b1)/sqrt(1 - b2) in the most severe sparse case, else ~ <= alpha.
Sec. 3: E[v_t] = E[g^2] (1 - b2^t) + zeta  ->  divide by (1 - b2^t).
3D toy: noisy, badly scaled quadratic f = 1/2 (20 x^2 + 0.05 y^2).
"""
from common import *

print("kingma2015adam")
B1, B2 = .9, .999


def adam_steps(gs, alpha=1., corr=True):
    m = v = 0.; out = []
    for t, g in enumerate(gs, 1):
        m = B1 * m + (1 - B1) * g; v = B2 * v + (1 - B2) * g * g
        mh, vh = (m / (1 - B1 ** t), v / (1 - B2 ** t)) if corr else (m, v)
        out.append(alpha * mh / (np.sqrt(vh) + 1e-12))
    return np.array(out)


T = 2000
const = adam_steps(np.full(T, 3.7))
check("bias-corrected Adam with a constant gradient steps exactly alpha", np.allclose(const, 1))
unc = adam_steps(np.full(T, 3.7), corr=False)
fac = (1 - B1 ** np.arange(1, T + 1)) / np.sqrt(1 - B2 ** np.arange(1, T + 1))
check("without correction the step is alpha (1 - b1^t)/sqrt(1 - b2^t)", np.allclose(unc, fac))
check("uncorrected steps peak near 6.5 alpha (t ~ 10) for b1 = 0.9, b2 = 0.999", abs(unc.max() - 6.5) < .3 and 5 <= np.argmax(unc) + 1 <= 15, f"max {unc.max():.2f} at t = {np.argmax(unc) + 1}")
for tt in [100, 1000]:
    sp = np.zeros(tt); sp[-1] = 5.
    st = adam_steps(sp)[-1]
    pred = (1 - B1) / np.sqrt(1 - B2) * np.sqrt(1 - B2 ** tt) / (1 - B1 ** tt)
    check(f"severe sparsity (one nonzero gradient at t = {tt}): step = alpha (1-b1)/sqrt(1-b2) sqrt(1-b2^t)/(1-b1^t)", abs(st - pred) < 1e-9, f"{st:.3f}")
check("bound (1 - b1)/sqrt(1 - b2) = 3.16 for the default settings", abs((1 - B1) / np.sqrt(1 - B2) - 3.1623) < 1e-3)
check("scale invariance: multiplying all gradients by c leaves the steps unchanged", np.allclose(adam_steps(np.random.default_rng(0).normal(size=200)), adam_steps(1000 * np.random.default_rng(0).normal(size=200))))
# 3D toy
A = np.array([20., .05])
f = lambda z: .5 * (A * z * z).sum(); g = lambda z: A * z
check_grad("toy quadratic", f, g, rand_pts(3, -2, 2))
SIG = .5


def run(kind, lr, T=600, seed=0):
    r = np.random.default_rng(seed); z = np.array([1.5, 1.8]); m = np.zeros(2); v = np.zeros(2); vel = np.zeros(2); out = []
    for t in range(1, T + 1):
        gr = g(z) + SIG * r.normal(size=2)
        if kind == "sgd": z = z - lr * gr
        elif kind == "mom": vel = .9 * vel - lr * gr; z = z + vel
        else:
            m = B1 * m + (1 - B1) * gr; v = B2 * v + (1 - B2) * gr * gr
            z = z - lr * (m / (1 - B1 ** t)) / (np.sqrt(v / (1 - B2 ** t)) + 1e-8)
        out.append(f(z))
    return np.array(out)


best = {}
for kind, grid in [("sgd", [.01, .03, .06, .09]), ("mom", [.003, .01, .03, .06, .09, .15]), ("adam", [.003, .01, .03, .1])]:
    sc = {lr: np.mean([run(kind, lr, seed=s)[-100:].mean() for s in range(4)]) for lr in grid}
    sc = {k: v for k, v in sc.items() if np.isfinite(v)}
    lr = min(sc, key=sc.get); best[kind] = (lr, sc[lr])
    print(f"    {kind}: best lr {lr}, late mean loss {sc[lr]:.4f}")
check("Adam reaches a lower late loss than tuned SGD on the badly scaled problem", best["adam"][1] < best["sgd"][1])
save_json("kingma2015adam", "data.json", {"best": {k: v[0] for k, v in best.items()}, "late": {k: round(float(v[1]), 4) for k, v in best.items()}})
done()
