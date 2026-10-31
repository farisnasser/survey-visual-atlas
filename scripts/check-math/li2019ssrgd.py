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
# in-page notes (papers/li2019ssrgd/notes.md): f1 = (x-1)^2/2, f2 = (x+1)^2/2, x0 = 2, v0 = 2, step 0.1
x1n = 2 - .1 * 2
dd = [(x1n - 1) - (2 - 1), (x1n + 1) - (2 + 1)]
check("notes example: x1 = 1.8, both differences -0.2, v1 = 1.8 = f'(1.8)", np.isclose(x1n, 1.8) and np.allclose(dd, -.2) and np.isclose(2 + dd[0], 1.8))
check("notes example: sample gradients at 1.8 are 0.8 and 2.8", np.allclose([x1n - 1, x1n + 1], [.8, 2.8]))
q = np.array([.01, .01])
for _ in range(10):
    q = q - .1 * np.array([q[0], -q[1]])
check("notes example: x10 ~ 0.00349, y10 ~ 0.02594", abs(q[0] - .00349) < 5e-6 and abs(q[1] - .02594) < 5e-6, f"{q}")
# primer idea 1: delta = sqrt(rho eps): sqrt(n)/delta^4 = sqrt(n)/(rho^2 eps^2), n/delta^3 = n/(rho eps)^1.5
for rho_, e_ in [(1, 1e-3), (3, 1e-2)]:
    dl = np.sqrt(rho_ * e_)
    assert np.isclose(1 / dl ** 4, 1 / (rho_ ** 2 * e_ ** 2)) and np.isclose(1 / dl ** 3, (rho_ * e_) ** -1.5)
check("primer: delta = sqrt(rho eps) turns sqrt(n)/delta^4 into sqrt(n)/eps^2 and n/delta^3 into n/eps^1.5", True)
# primer idea 2: n = 1e6, eps = 1e-3: sqrt(n)/eps^2 = 1e9 vs n^(2/3)/eps^2 = 1e10
check("primer: 1e9 vs 1e10", np.isclose(1e3 / 1e-6, 1e9) and np.isclose(1e6 ** (2 / 3) / 1e-6, 1e10))
# primer idea 3: ln(50)/ln(1.1) ~ 41 steps from 0.01 to 0.5
yy, kk = .01, 0
while yy < .5:
    yy *= 1.1; kk += 1
check("primer: ln 50 / ln 1.1 ~ 41 steps from 0.01 to 0.5", round(np.log(50) / np.log(1.1)) == 41 and kk == 42, f"{np.log(50) / np.log(1.1):.2f}, first step beyond 0.5: {kk}")
# primer idea 4 (paper Sec. 5.2): f_thres/t_thres = (delta^3/rho^2)/(1/(eta delta)) = delta^4/(L rho^2) at eta = 1/L; b + n/m = 2 sqrt(n)
for L_, rho_, dl, n_ in [(1, 1, .1, 1e4), (5, 2, .3, 1e6)]:
    assert np.isclose((dl ** 3 / rho_ ** 2) / (L_ / dl), dl ** 4 / (L_ * rho_ ** 2)) and np.isclose(np.sqrt(n_) + n_ / np.sqrt(n_), 2 * np.sqrt(n_))
    D_ = 1.
    assert np.isclose(2 * np.sqrt(n_) * D_ / (dl ** 4 / (L_ * rho_ ** 2)), 2 * L_ * rho_ ** 2 * D_ * np.sqrt(n_) / dl ** 4)
check("primer: per-iteration decrease delta^4/(L rho^2) at cost 2 sqrt(n) gives L rho^2 D sqrt(n)/delta^4", True)
done()
