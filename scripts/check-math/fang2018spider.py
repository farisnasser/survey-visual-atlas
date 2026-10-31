"""Fang, Li, Lin, Zhang 2018 (SPIDER): path-integrated gradient estimator and Lemma 1, on a real finite sum.

f_i(x, y) = (x^2 - 1)^2/4 + y^2/2 + a_i x + b_i y + c_i (x^2 - y^2)/2,  sum a = sum b = sum c = 0  =>  f = mean f_i.
SPIDER: every q steps v = grad f (full), else v_k = grad f_S(x_k) - grad f_S(x_{k-1}) + v_{k-1} (same minibatch S at both points).
SFO Option I: x_{k+1} = x_k - eta v_k/|v_k| (so |x_k - x_{k-1}| = eta = eps_1 of Lemma 1 exactly).
"""
from common import *

print("fang2018spider")
n = 200
R0 = np.random.default_rng(4)
a = R0.normal(0, 1.2, n); b = R0.normal(0, 1.2, n); c = R0.normal(0, .8, n)
a -= a.mean(); b -= b.mean(); c -= c.mean()
f = lambda p: .25 * (p[0] ** 2 - 1) ** 2 + .5 * p[1] ** 2
g = lambda p: np.array([p[0] ** 3 - p[0], p[1]])
fi = lambda p, i: f(p) + a[i] * p[0] + b[i] * p[1] + c[i] * (p[0] ** 2 - p[1] ** 2) / 2
gi = lambda p, I: np.stack([g(p)[0] + a[I] + c[I] * p[0], g(p)[1] + b[I] - c[I] * p[1]], -1)
check_grad("f", f, g, rand_pts(20, -1.8, 1.8))
check_grad("f_7", lambda p: fi(p, 7), lambda p: gi(p, np.array([7]))[0], rand_pts(20, -1.8, 1.8))
check("mean of component gradients = grad f", np.allclose(gi(np.array([.3, -.7]), np.arange(n)).mean(0), g([.3, -.7])))
# mean-square smoothness constant on the box |x| <= 1.8: E|grad f_i(p) - grad f_i(q)|^2 <= L^2 |p - q|^2
L = np.sqrt(np.mean(np.maximum((3 * 1.8 ** 2 - 1 + np.abs(c)) ** 2, (1 + np.abs(c)) ** 2)))
print(f"  mean-square smoothness L = {L:.3f}")
eta, q, S2 = .02, 25, 4


def run(K, seed, est="spider"):
    R = np.random.default_rng(seed)
    x = np.array([-1.6, 1.4]); xp = x.copy(); v = None; err = []
    for k in range(K):
        if est == "spider":
            if k % q == 0:
                v = g(x)
            else:
                I = R.integers(0, n, S2)
                v = (gi(x, I) - gi(xp, I)).mean(0) + v
        else:
            I = R.integers(0, n, S2); v = gi(x, I).mean(0)
        err.append(np.sum((v - g(x)) ** 2))
        xp = x.copy(); x = x - eta * v / max(np.linalg.norm(v), 1e-12)
    return np.array(err), x


E = np.mean([run(q, s)[0] for s in range(400)], 0)
bound = np.arange(q) * L ** 2 * eta ** 2 / S2
check("Lemma 1: E|v_k - grad f(x_k)|^2 <= k L^2 eps1^2 / S2 within each epoch", np.all(E <= bound + 1e-12), f"at k = {q - 1}: {E[-1]:.2e} <= {bound[-1]:.2e}")
check("error grows roughly linearly in k (martingale sum)", E[-1] > 5 * E[3] > 0)
Es = np.mean([run(q, s, "sgd")[0] for s in range(400)], 0)
check("plain minibatch SGD estimator error is much larger than SPIDER's", Es.mean() > 20 * E.mean(), f"SGD {Es.mean():.3f} vs SPIDER {E.mean():.4f}")
err, xe = run(600, 0)
check("SPIDER-SFO reaches a minimum neighbourhood (|grad f| ~ eta)", np.linalg.norm(g(xe)) < 5 * eta and abs(abs(xe[0]) - 1) < .1, f"x = {np.round(xe, 3)}")
save_json("fang2018spider", "data.json", {"a": r(a, 5), "b": r(b, 5), "c": r(c, 5), "L": round(float(L), 4)})

# in-page notes (papers/fang2018spider/notes.md): shared-noise cancellation (Sec. 3) and primer numbers
rngN = np.random.default_rng(11)
h_ = lambda x: np.sin(x) + x ** 2
xi_ = rngN.normal(0, 3, 1000)
gi = lambda x, i: h_(x) + xi_[i]
check("notes: g_i(x_t) - g_i(x_{t-1}) = h(x_t) - h(x_{t-1}) exactly for g_i = h + xi_i",
      all(abs((gi(.71, i) - gi(.70, i)) - (h_(.71) - h_(.70))) < 1e-12 for i in range(1000)))
check("primer: sigma^2 = 4, b = 16 -> minibatch variance 0.25", abs(4 / 16 - .25) < 1e-15)
mb = rngN.normal(0, 2, (200000, 16)).mean(1)
check("primer: Monte Carlo minibatch variance ~ 0.25", abs(mb.var() - .25) < .005, f"{mb.var():.4f}")
c_ = rngN.normal(1, 1, 400000)          # Var(c_i) = 1
dif = c_ * (.51 - .50)
check("primer: Var(c_i (x_t - x_{t-1})) = Var(c) * 0.01^2 = 1e-4", abs(dif.var() - 1e-4) < 3e-6, f"{dif.var():.3e}")
check("primer: Lemma 1 bound k L^2 eps1^2 / S2 = 1e-3 (std ~ 0.032) for L=1, eps1=0.01, S2=10, k=100",
      abs(100 * 1 * .01 ** 2 / 10 - 1e-3) < 1e-15 and abs(np.sqrt(1e-3) - .032) < 5e-4)
# Monte Carlo of the running SPIDER estimate on f_i = c_i x^2/2 with steps of 0.01 and S2 = 10 (L^2 = E c^2 = 2 here)
K, S2_, reps = 100, 10, 4000
err = np.zeros(reps)
for r in range(reps):
    cs = rngN.normal(1, 1, (K, S2_))
    err[r] = np.sum(cs.mean(1) * .01 - 1 * .01)   # sum of per-step estimator errors (true curvature mean 1)
check("primer: estimator error variance after k steps <= k L^2 eps1^2 / S2 (L^2 = E c^2 = 2)", err.var() <= K * 2 * .01 ** 2 / S2_, f"{err.var():.2e} <= {K * 2 * 1e-4 / S2_:.2e}")
e_ = 1e-2
check("primer: eps = 0.01, n = 100: SPIDER 1e5 vs GD 1e6", abs(min(np.sqrt(100) * e_ ** -2, e_ ** -3) - 1e5) < 1e-6 and abs(100 * e_ ** -2 - 1e6) < 1e-6)
check("primer: eps = 0.01, n = 1e6: SPIDER 1e6 (online term) vs GD 1e10", abs(min(np.sqrt(1e6) * e_ ** -2, e_ ** -3) - 1e6) < 1e-3 and abs(1e6 * e_ ** -2 - 1e10) < 1)
check("primer: sqrt(n) eps^-2 = eps^-3 at n = eps^-2", abs(np.sqrt(e_ ** -2) * e_ ** -2 - e_ ** -3) < 1e-6)
done()
