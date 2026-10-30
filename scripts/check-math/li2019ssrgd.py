"""Li 2019 (SSRGD, Algorithm 1): SARAH/SPIDER recursive gradient + a uniform perturbation when the full gradient is small.

Finite sum f = mean f_i, f_i(x, y) = (x^2 - 1)^2/4 + y^2/2 + a_i x + b_i y + c_i (x^2 - y^2)/2 (zero-mean a, b, c).
Start (0, 1.3) on the stable manifold x = 0 of the saddle (0, 0): every component's x-gradient is a_i + (c_i - 1) x + x^3,
so with x = 0 the recursive differences have x-part exactly 0 and plain SARAH converges to the saddle.
SSRGD: at an epoch start, if |grad f(x_sm)| <= g_thres and not in a super epoch: x += xi ~ Unif(B(r)), start a super epoch;
it ends when the iterate has moved more than R_stop from the perturbation point (simplified stop rule) or after t_thres steps.
"""
from common import *

print("li2019ssrgd")
n = 196
m = b = int(np.sqrt(n))   # 14 (Theorem 1: m = b = sqrt(n))
R0 = np.random.default_rng(83)
a = R0.normal(0, 1.0, n); bb = R0.normal(0, 1.0, n); c = R0.normal(0, .6, n)
a -= a.mean(); bb -= bb.mean(); c -= c.mean()
f = lambda p: .25 * (p[0] ** 2 - 1) ** 2 + .5 * p[1] ** 2
g = lambda p: np.array([p[0] ** 3 - p[0], p[1]])
gi = lambda p, I: np.stack([p[0] ** 3 - p[0] + a[I] + c[I] * p[0], p[1] + bb[I] - c[I] * p[1]], -1)
fi = lambda p, i: f(p) + a[i] * p[0] + bb[i] * p[1] + c[i] * (p[0] ** 2 - p[1] ** 2) / 2
check_grad("f_9", lambda p: fi(p, 9), lambda p: gi(p, np.array([9]))[0], rand_pts(20, -1.6, 1.6))
check("mean component gradient = grad f", np.allclose(gi(np.array([.2, -.5]), np.arange(n)).mean(0), g([.2, -.5])))
L = float(3 * 1.6 ** 2 - 1 + np.max(np.abs(c)))
eta = (np.sqrt(5) - 1) / (2 * L)
print(f"  L = {L:.3f} on |x| <= 1.6, eta = (sqrt5 - 1)/(2L) = {eta:.4f}, m = b = {m}")
gthres, r_, Rstop, tthres = .02, .01, .3, 400
x0 = np.array([0., 1.3])


def run(seed, perturb, K=2500):
    R = np.random.default_rng(seed)
    x = x0.copy(); sfo = 0; inside = False; xs = None; ts = 0; t = 0; out = [(0, x[0], x[1])]; kicks = []
    while t < K:
        gf = g(x); sfo += n
        if perturb and not inside and np.linalg.norm(gf) <= gthres:
            th = R.uniform(0, 2 * np.pi); rad = r_ * np.sqrt(R.uniform()); x = x + rad * np.array([np.cos(th), np.sin(th)])
            inside, xs, ts = True, x.copy(), t; kicks.append(t); gf = g(x)
        v = gf
        for k in range(m):
            xp = x.copy(); x = x - eta * v; t += 1
            I = R.integers(0, n, b); v = (gi(x, I) - gi(xp, I)).mean(0) + v; sfo += 2 * b
            out.append((sfo, x[0], x[1]))
            if inside and (np.linalg.norm(x - xs) > Rstop or t - ts > tthres):
                inside = False
    return np.array(out), x, kicks


O, xs_, _ = run(0, False)
check("plain SARAH (no perturbation) from the stable manifold keeps x exactly 0 and converges to the saddle", np.all(O[:, 1] == 0) and abs(xs_[1]) < 1e-6, f"x = {xs_}")
res = [run(s, True) for s in range(50)]
fin = [np.linalg.norm(g(r__[1])) < .02 and abs(abs(r__[1][0]) - 1) < .05 for r__ in res]
check("SSRGD reaches a minimum (+-1, 0) in every one of 50 runs", all(fin))
nk = [len(r__[2]) for r__ in res]
print(f"  perturbations per run: median {np.median(nk):.0f}, max {max(nk)}")
sfo_hit = [r__[0][np.argmax(np.abs(r__[0][:, 1]) > .9), 0] for r__ in res]
print(f"  SFO calls until |x| > 0.9: median {np.median(sfo_hit):.0f}")
check("the notes' 2-D example: x_10 = 0.01*0.9^10 = 0.00349, y_10 = 0.01*1.1^10 = 0.02594", abs(.01 * .9 ** 10 - .00349) < 5e-6 and abs(.01 * 1.1 ** 10 - .02594) < 5e-6)
save_json("li2019ssrgd", "data.json", {"a": r(a, 5), "b": r(bb, 5), "c": r(c, 5), "L": round(L, 5), "eta": round(float(eta), 6), "m": m,
                                      "gthres": gthres, "r": r_, "Rstop": Rstop, "tthres": tthres, "sfo_med": float(np.median(sfo_hit))})
done()
