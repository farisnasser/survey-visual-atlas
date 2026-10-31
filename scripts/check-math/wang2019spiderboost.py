"""Wang, Ji, Zhou, Liang, Tarokh 2019 (SpiderBoost): SPIDER's estimator with a constant step 1/(2L), and the prox variant.

Finite sum f = mean f_i, f_i(x, y) = (x^2 - 1)^2/4 + (y - 0.25)^2/2 + a_i x + b_i y + c_i (x^2 - y^2)/2, zero-mean a, b, c.
Estimator (Eq. 1): v_k = grad f(x_k) every q steps, else v_k = mean_{i in S} [grad f_i(x_k) - grad f_i(x_{k-1})] + v_{k-1}.
SPIDER-SFO: x+ = x - (eps/L) v/|v| (normalized, eps-level).  SpiderBoost (Alg. 1): x+ = x - v/(2L).  q = |S| = sqrt(n).
Prox-SpiderBoost (Alg. 2): x+ = prox_{eta h}(x - eta v) with h = 0.3(|x| + |y|) (soft-thresholding).
"""
from common import *

print("wang2019spiderboost")
n = 196
q = S = int(np.sqrt(n))
R0 = np.random.default_rng(80)
a = R0.normal(0, 1.2, n); b = R0.normal(0, 1.2, n); c = R0.normal(0, .8, n)
a -= a.mean(); b -= b.mean(); c -= c.mean()
f = lambda p: .25 * (p[0] ** 2 - 1) ** 2 + .5 * (p[1] - .25) ** 2
g = lambda p: np.array([p[0] ** 3 - p[0], p[1] - .25])
gi = lambda p, I: np.stack([p[0] ** 3 - p[0] + a[I] + c[I] * p[0], p[1] - .25 + b[I] - c[I] * p[1]], -1)
fi = lambda p, i: f(p) + a[i] * p[0] + b[i] * p[1] + c[i] * (p[0] ** 2 - p[1] ** 2) / 2
check_grad("f", f, g, rand_pts(20, -1.8, 1.8))
check_grad("f_5", lambda p: fi(p, 5), lambda p: gi(p, np.array([5]))[0], rand_pts(20, -1.8, 1.8))
check("mean of component gradients = grad f", np.allclose(gi(np.array([.4, -.3]), np.arange(n)).mean(0), g([.4, -.3])))
# Assumption 1: every grad f_i is L-Lipschitz (on the box |x| <= 1.8): Hessian of f_i = diag(3x^2 - 1 + c_i, 1 - c_i)
L = float(max(np.max(np.abs(3 * 1.8 ** 2 - 1 + c)), np.max(np.abs(1 - c)), np.max(np.abs(-1 + c))))
print(f"  individual smoothness L = {L:.3f} on |x| <= 1.8, q = |S| = {q}")
x0 = np.array([-1.6, 1.4]); EPS = .05


def run(kind, K, seed, prox=0.0):
    R = np.random.default_rng(seed)
    x = x0.copy(); xp = x.copy(); v = None; sfo = 0; out = [(0, np.linalg.norm(g(x)), f(x))]
    for k in range(K):
        if k % q == 0:
            v = g(x); sfo += n
        else:
            I = R.integers(0, n, S); v = (gi(x, I) - gi(xp, I)).mean(0) + v; sfo += 2 * S
        xp = x.copy()
        if kind == "spider":
            x = x - EPS / L * v / max(np.linalg.norm(v), 1e-12)
        elif kind == "boost":
            x = x - v / (2 * L)
        elif kind == "prox":
            z = x - v / (2 * L); x = np.sign(z) * np.maximum(np.abs(z) - prox / (2 * L), 0)
        elif kind == "subgrad":
            x = x - (v + prox * np.sign(x)) / (2 * L)
        out.append((sfo, np.linalg.norm(g(x)), f(x)))
    return np.array(out), x


B, xb = run("boost", 400, 1)
Sp, xs = run("spider", 4000, 1)
hit = lambda o, t: o[np.argmax(o[:, 1] < t), 0] if np.any(o[:, 1] < t) else np.inf
hb, hs = hit(B, EPS), hit(Sp, EPS)
check("SpiderBoost reaches |grad f| < eps with at least 3x fewer SFO calls than SPIDER", hb * 3 < hs, f"SFO to |grad| < {EPS}: SpiderBoost {hb:.0f}, SPIDER {hs:.0f}")
check("SpiderBoost ends near a minimum (+-1, 0.25)", abs(abs(xb[0]) - 1) < .02 and abs(xb[1] - .25) < .02, f"x = {np.round(xb, 4)}")
# Theorem 1 is about E|grad f(x_xi)| for a random iterate; over 50 seeds, the average over iterates after the first epoch
Es = np.mean([np.mean(run("boost", 200, s)[0][q:, 1]) for s in range(50)])
check("mean |grad f| over iterates is small (random-output guarantee)", Es < .1, f"{Es:.4f}")

# prox: h(x, y) = lam (|x| + |y|), minimizers of psi = f + h: y* = 0 (since lam > 0.25), x^3 - x + lam sign(x) = 0
lam = .3
r3 = np.roots([1, 0, -1, lam])  # x > 0 branch: x^3 - x + lam = 0; the largest root is the minimizer
xs_ = float(max(rr.real for rr in r3 if abs(rr.imag) < 1e-12))
print(f"  composite minimizers: (+-{xs_:.4f}, 0)")
P, xpz = run("prox", 600, 2, lam)
Sg, xsg = run("subgrad", 600, 2, lam)
check("Prox-SpiderBoost lands with y exactly 0 (soft-thresholding)", xpz[1] == 0.0 and abs(abs(xpz[0]) - xs_) < .005, f"x = {xpz}")
ys = [run("subgrad", 600, s, lam)[1][1] for s in range(10)]
check("plain subgradient steps never make y exactly 0", all(y != 0 for y in ys), f"|y| ~ {np.mean(np.abs(ys)):.1e}")
check("soft-threshold example from the notes: prox of 0.3|x| at 0.8 with eta = 1 is 0.5", abs(np.sign(.8) * max(.8 - .3, 0) - .5) < 1e-12)
save_json("wang2019spiderboost", "data.json", {"a": r(a, 5), "b": r(b, 5), "c": r(c, 5), "L": round(L, 5), "q": q, "eps": EPS, "lam": lam,
                                              "xs": round(xs_, 5), "hb": float(hb), "hs": float(hs)})
# in-page notes (papers/wang2019spiderboost/notes.md): f1 = (x-1)^2/2, f2 = (x+1)^2/2, x_{k-1} = 1.5, v_{k-1} = 1.5, x_k = 2
d1 = lambda x: x - 1; d2 = lambda x: x + 1
diffs = [d1(2) - d1(1.5), d2(2) - d2(1.5)]
check("notes example: f_i'(2) - f_i'(1.5) = 0.5 for both samples, v_k = 2 = f'(2)", np.allclose(diffs, .5) and all(np.isclose(1.5 + q, 2) for q in diffs))
check("notes example: raw sample gradients at x = 2 are 1 and 3", (d1(2), d2(2)) == (1, 3))
soft = lambda z, t: np.sign(z) * max(abs(z) - t, 0)
xs = np.linspace(-2, 2, 400001)
check("notes: prox of 0.3|x| at z = 0.8 (eta = 1) is 0.5", np.isclose(soft(.8, .3), .5) and abs(xs[np.argmin(.3 * abs(xs) + (xs - .8) ** 2 / 2)] - .5) < 1e-4)
# primer idea 1: n eps^-2 = 1e10 vs n + sqrt(n) eps^-2 = 1.1e7 (n = 1e6, eps = 1e-2)
nn, ee = 1e6, 1e-2
check("primer: n/eps^2 = 1e10, n + sqrt(n)/eps^2 = 1.1e7", np.isclose(nn / ee ** 2, 1e10) and np.isclose(nn + np.sqrt(nn) / ee ** 2, 1.1e7))
# primer idea 2: at x = 0 the sample gradients are -1, +1 (variance 1), full gradient 0
check("primer: sample gradients at 0 are -1, +1 with variance 1", (d1(0), d2(0)) == (-1, 1) and np.isclose(np.var([d1(0), d2(0)]), 1))
# primer idea 3: epoch cost n + (q-1) 2|S| with q = |S| = sqrt(n) = 1000 -> ~3.0e6 over 1000 iterations ~ 3 sqrt(n)
ep = nn + 999 * 2 * 1000
check("primer: epoch cost ~3.0e6 SFO, ~3 sqrt(n) per iteration", np.isclose(ep, 2.998e6) and abs(ep / 1000 / np.sqrt(nn) - 3) < .01)
# primer idea 4: Psi = (x-1)^2/2 + 0.3|x|, minimizer 0.7; G_1(x) = x - prox(x - f'(x))
G = lambda x: x - soft(x - (x - 1), .3)
Ps = (xs - 1) ** 2 / 2 + .3 * abs(xs)
check("primer: argmin Psi = 0.7, G(0.7) = 0, G(0) = -0.7", abs(xs[np.argmin(Ps)] - .7) < 1e-4 and np.isclose(G(.7), 0) and np.isclose(G(0), -.7))
done()
