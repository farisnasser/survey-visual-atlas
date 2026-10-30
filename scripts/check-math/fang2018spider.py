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
done()
