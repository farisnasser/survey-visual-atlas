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

# in-page notes (papers/kingma2015adam/notes.md): Sec. 3 bias-correction algebra and the primer's numbers
b1, b2 = .9, .999
for gc in (.3, -2.0):
    m = v = 0.0
    for t in range(1, 30):
        m = b1 * m + (1 - b1) * gc; v = b2 * v + (1 - b2) * gc * gc
        assert np.isclose(m, (1 - b1 ** t) * gc) and np.isclose(m / (1 - b1 ** t), gc) and np.isclose(v / (1 - b2 ** t), gc * gc)
check("notes Sec. 3: m_t = (1 - b1^t) g, m_hat = g, v_hat = g^2 for a constant gradient", True)
check("notes Sec. 3: m_2 = (1-b1)(1+b1) g", np.isclose(b1 * (1 - b1) + (1 - b1), (1 - b1) * (1 + b1)))
check("primer idea 1: memory ~ 1/(1-beta) = 10 and 1000; 0.9^10 ~ 0.35", np.isclose(1 / (1 - b1), 10) and np.isclose(1 / (1 - b2), 1000) and round(.9 ** 10, 2) == .35)
Rk = np.random.default_rng(51)
noise = Rk.normal(0, 1, 400000); mm = 0.0; tail = []
for i, e in enumerate(noise):
    mm = .9 * mm + .1 * e
    if i > 1000: tail.append(mm)
check("primer idea 2: EMA noise variance sigma^2 (1-b)/(1+b) = sigma^2/19 at b = 0.9", abs(np.var(tail) - 1 / 19) < 3e-3 and np.isclose(.1 / 1.9, 1 / 19), f"{np.var(tail):.4f} vs {1 / 19:.4f}")
def adam_step(gseq, a=1.0):
    m = v = 0.0
    for t, gg in enumerate(gseq, 1):
        m = b1 * m + (1 - b1) * gg; v = b2 * v + (1 - b2) * gg * gg
    return a * (m / (1 - b1 ** t)) / np.sqrt(v / (1 - b2 ** t))
check("primer idea 3: constant gradients 0.001 and 1000 both give step alpha", np.isclose(adam_step([1e-3] * 50), 1) and np.isclose(adam_step([1e3] * 50), 1))
signs = Rk.choice([-1.0, 1.0], 3000)
check("primer idea 4: +-1 random gradients give a small step, constant +1 gives alpha", abs(adam_step(signs)) < .5 and np.isclose(adam_step([1.0] * 3000), 1))
unc = [(1 - b1 ** t) / np.sqrt(1 - b2 ** t) for t in range(1, 200)]
check("primer idea 5: uncorrected step 3.16 alpha at t=1, peak ~6.57 alpha at t=12",
      round(unc[0], 2) == 3.16 and int(np.argmax(unc)) + 1 == 12 and round(max(unc), 2) == 6.57, f"t=1: {unc[0]:.3f}, max {max(unc):.3f} at t={int(np.argmax(unc)) + 1}")
done()
