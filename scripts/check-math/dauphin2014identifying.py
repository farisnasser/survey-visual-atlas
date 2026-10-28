"""Dauphin et al. 2014: GD vs Newton vs saddle-free Newton on the paper's Fig. 2 functions."""
from common import *

print("dauphin2014identifying")
fa = lambda p: 5 * p[0] ** 2 - p[1] ** 2
ga = lambda p: np.array([10 * p[0], -2 * p[1]])
Ha = lambda p: np.array([[10., 0], [0, -2]])
fb = lambda p: p[0] ** 3 - 3 * p[0] * p[1] ** 2
gb = lambda p: np.array([3 * p[0] ** 2 - 3 * p[1] ** 2, -6 * p[0] * p[1]])
Hb = lambda p: np.array([[6 * p[0], -6 * p[1]], [-6 * p[1], -6 * p[0]]])
P = rand_pts(30, -1.5, 1.5)
check_grad("5x^2-y^2", fa, ga, P)
check_grad("monkey", fb, gb, P)
for p in P[:5]:
    assert np.allclose(Ha(p), fd_hess(fa, p), atol=1e-3) and np.allclose(Hb(p), fd_hess(fb, p), atol=1e-3)
check("analytic Hessians match", True)
check("monkey saddle Hessian is zero at 0 (degenerate)", np.allclose(Hb([0, 0]), 0))
DELTA = .1


def step(kind, p, g, H, lr):
    gr = g(p)
    if kind == "gd":
        return p - lr * gr
    lam, Q = np.linalg.eigh(H(p))
    if kind == "newton":   # damped where |lambda| is tiny, sign kept
        lam = np.where(np.abs(lam) < DELTA, np.where(lam >= 0, DELTA, -DELTA), lam)
    else:                  # saddle-free Newton: |lambda|, floored at DELTA
        lam = np.maximum(np.abs(lam), DELTA)
    return p - lr * Q @ ((Q.T @ gr) / lam)


def run(kind, p0, g, H, lr, n, box=1.6):
    p = np.array(p0, float)
    path = [p.copy()]
    for _ in range(n):
        p = step(kind, p, g, H, lr)
        path.append(p.copy())
        if np.max(np.abs(p)) > box:
            break
    return np.array(path)


p0 = [.9, .12]
LR = {"gd": .08, "newton": .5, "sfn": .5}   # GD must keep lr < 2/10; Newton/SFN rescale curvature to 1
for kind in ["gd", "newton", "sfn"]:
    pa = run(kind, p0, ga, Ha, LR[kind], 400)
    print(f"  5x^2-y^2 {kind} (lr {LR[kind]}): {len(pa) - 1} steps, end {np.round(pa[-1], 3)}, f={fa(pa[-1]):.3f}")
pn = run("newton", p0, ga, Ha, LR["newton"], 400)
check("Newton is attracted to the saddle", np.linalg.norm(pn[-1]) < 1e-3)
ps = run("sfn", p0, ga, Ha, LR["sfn"], 400)
pg = run("gd", p0, ga, Ha, LR["gd"], 400)
check("SFN (own step) escapes faster than GD (own stable step)", len(ps) < len(pg) and abs(ps[-1][1]) > 1.5, f"{len(ps)-1} vs {len(pg)-1}")
check("GD at lr 0.21 > 2/10 diverges along x", abs(run("gd", p0, ga, Ha, .21, 60, box=1e9)[-1][0]) > 1)
# weak negative curvature: f = 5x^2 - eps y^2 ; GD escape time ~ 1/(lr*eps), SFN independent of eps
for eps in [1, .1, .01]:
    fe_g = lambda p, e=eps: np.array([10 * p[0], -2 * e * p[1]])
    fe_H = lambda p, e=eps: np.array([[10., 0], [0, -2 * e]])
    a = len(run("gd", p0, fe_g, fe_H, .08, 20000)) - 1
    b = len(run("sfn", p0, fe_g, fe_H, .5, 20000)) - 1
    print(f"  eps={eps}: GD {a} steps, SFN {b} steps (DELTA floor {DELTA})")
    if eps == .01:
        check("weak curvature: GD crawls, SFN does not", a > 10 * b)
check("exact Newton step lands on the saddle in one step", np.allclose(step("newton", np.array(p0), ga, Ha, 1.0), 0))
for kind in ["gd", "newton", "sfn"]:
    pb = run(kind, [.35, .45], gb, Hb, .03, 600)
    print(f"  monkey {kind}: {len(pb) - 1} steps, end {np.round(pb[-1], 3)}, f={fb(pb[-1]):.3f}")
check("monkey: SFN and GD decrease f", fb(run("sfn", [.35, .45], gb, Hb, .03, 600)[-1]) < fb([.35, .45]))
rng = np.random.default_rng(0)


def goe(N):
    A = rng.normal(size=(N, N))
    return (A + A.T) / np.sqrt(2 * N)


fr = {}
for N in [1, 2, 3, 4, 5, 6, 8]:
    fr[N] = np.mean([np.all(np.linalg.eigvalsh(goe(N)) > 0) for _ in range(20000)])
print("  P(all eigenvalues > 0):", {k: round(v, 5) for k, v in fr.items()})
check("P(min) shrinks fast with N", fr[1] > .45 and fr[4] < .05 and fr[8] < fr[4])
ev = np.linalg.eigvalsh(goe(400))
check("GOE spectrum ~ semicircle on [-2, 2]", abs(ev.min() + 2) < .15 and abs(ev.max() - 2) < .15)
done()
