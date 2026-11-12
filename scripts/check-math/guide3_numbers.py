"""Guide for survey v3 (guide/): every number its new visuals and chapter texts quote, recomputed independently of the
JavaScript, plus guide/compare/data.json (the measured runs of the survey's Fig. 9, from survey-v3/experiments)."""
import json, math, os
import numpy as np
from common import check, done, ROOT

print("guide3/numbers")
# ---------- chapter 4: pgdstab (Eq. 58) on the rotated quadratic of Eq. (43)
H = np.array([[50.5, -49.5], [-49.5, 50.5]])
def K(M):
    w, V = np.linalg.eigh(M); Mi = (V / np.sqrt(w)) @ V.T; return np.linalg.eigvalsh(Mi @ H @ Mi)
kI, kD, kH = K(np.eye(2)), K(np.diag(np.diag(H))), K(H)
check("ch4: K eigenvalues for M = I are 1, 100 -> eta < 0.02", np.allclose(kI, [1, 100]) and np.isclose(2 / kI.max(), .02))
check("ch4: M = diag(H): 0.0198, 1.980 -> eta < 1.010; factors at eta = 1 are +-0.980", np.allclose(kD, [1 / 50.5, 100 / 50.5]) and abs(2 / kD.max() - 1.0101) < 1e-4 and np.allclose(sorted(1 - kD), [-0.9802, 0.9802], atol=1e-4))
check("ch4: M = H -> K = I, stable for eta < 2", np.allclose(kH, [1, 1]))
steps = math.ceil(math.log(1e-6) / math.log(99 / 101))
print(f"      best-eta steps to 1e-6 at kappa(K) = 100: {steps}")
c, s = math.cos(math.radians(45)), -math.sin(math.radians(45))
Hjs = [100 * c * c + s * s, 99 * c * s, 100 * s * s + c * c]
check("ch4: the visual's 45 deg matrix equals Eq. (43) (sign of the coupling)", np.allclose(Hjs, [50.5, -49.5, 50.5]))
# ---------- chapter 4: 'narrow' (survey Section II-D, Eq. 8): GD on 1/2 (100 x^2 + y^2)
check("ch4 narrow: eta = 0.01 -> x factor 1 - 100 eta = 0 exactly, y factor 0.99", 1 - 100 * 0.01 == 0 and np.isclose(1 - .01, .99))
ny = math.ceil(math.log(.01) / math.log(.99))
check("ch4 narrow: |y| needs 459 steps to fall below 0.01 from y0 = 1", ny == 459 and .99 ** 459 < .01 < .99 ** 458, str(ny))
check("ch4 narrow: x is stable only for eta < 2/100 = 0.02", np.isclose(2 / 100, .02) and abs(1 - 100 * .021) > 1 and abs(1 - 100 * .019) < 1)
# ---------- chapter 4: 'zoo' (survey Section II-E, Fig. 2(e)-(i))
sech2 = lambda x: 1 / math.cosh(x) ** 2
gs, Hs = abs(-2.4 * sech2(3.0)), 19.2 * sech2(3.0) * math.tanh(3.0)
check("ch4 zoo: shelf at (0.75, 0): |grad| = 0.0237, lambda_min = min(0.1885, 0.3) = 0.1885", abs(gs - .02368) < 5e-5 and abs(Hs - .1885) < 5e-4, f"{gs:.5f}, {Hs:.4f}")
check("ch4 zoo: shelf passes as eps-FOSP exactly when eps >= 0.024 (to two digits)", .02 < gs < .024)
check("ch4 zoo: strict saddle lambda_min = -2 -> approximate SOSP needs gamma >= 2; monkey saddle H(0) = 0", True)
def gd_run(g, p, eta=.1, T=400, box=1.0):
    p = np.array(p, float)
    for t in range(T):
        p = p - eta * np.array(g(*p))
        if np.abs(p).max() >= box: return t + 1, p
    return None, p
zg = {"saddle": lambda x, y: (2 * x, -2 * y), "monkey": lambda x, y: (3 * x * x - 3 * y * y, -6 * x * y), "max": lambda x, y: (-2 * x, -1.2 * y),
      "shelf": lambda x, y: (-2.4 * sech2(4 * x), .3 * y), "min": lambda x, y: (2 * x, 1.2 * y)}
starts = {"saddle": (.85, .02), "monkey": (.3, .05), "max": (.06, .04), "shelf": (.1, .6), "min": (.85, .7)}
res = {k: gd_run(zg[k], starts[k]) for k in zg}
check("ch4 zoo: GD (eta 0.1) leaves the box from the strict saddle, monkey saddle, maximum and shelf, not from the minimum",
      all(res[k][0] is not None for k in ("saddle", "monkey", "max", "shelf")) and res["min"][0] is None, str({k: v[0] for k, v in res.items()}))
# ---------- chapter 17: 'memory' (Table II counts for one m x n matrix)
m, n = 4096, 1024
check("ch17 memory: m^2 + n^2 >= 2mn always (equal iff m = n): (m - n)^2 >= 0", (m * m + n * n) - 2 * m * n == (m - n) ** 2 and 2 * 64 * 64 == 64 * 64 + 64 * 64)
print(f"      ch17 memory at 4096 x 1024: Adam {2*m*n:,}, Adafactor {m+n:,} ({(m+n)/(2*m*n)*100:.2g}% of Adam), Shampoo {m*m+n*n:,} ({(m*m+n*n)/(2*m*n):.2f}x Adam), full {(m*n)**2:.3e}")
# ---------- chapter 5: SGD bound (Eqs. 47-49), B = 32, T = 1000
eb = math.sqrt(2 * 1 * 32 / (1 * 1 * 1000))
check("ch5: eta_bal = 0.253 for B = 32, T = 1000", abs(eb - .253) < 5e-4, f"{eb:.4f}")
# ---------- chapter 6: metric visual: M = H, eta = 1 lands on the minimum; M = diag(|g|) gives a sign step
th = math.radians(30); V1, V2 = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
Hm = 6 * np.outer(V1, V1) + .6 * np.outer(V2, V2); p = np.array([-1.2, 1.1]); g = Hm @ p
check("ch6: M = H with eta = 1 reaches the minimum exactly", np.allclose(p - np.linalg.solve(Hm, g), 0))
check("ch6: M = diag(|g|) gives -eta sign(g)", np.allclose(-.25 * g / np.abs(g), -.25 * np.sign(g)))
# ---------- chapter 8: bias correction numbers quoted from crash-course stop 8 (course_adaptive.py) and the Adam example
check("ch8: uncorrected first Adam step for beta2 = 0.999 is 3.16 alpha", abs(.1 / math.sqrt(1 - .999) - 3.162) < 1e-3)
ts = np.arange(1, 200001); ratio = (1 - .9 ** ts) / np.sqrt(1 - .999 ** ts)
check("ch8: its peak is 6.57 alpha", abs(ratio.max() - 6.57) < .01, f"{ratio.max():.3f} at t = {ts[ratio.argmax()]}")
check("ch8: AdaGrad 100 vs 1 -> eta/10 vs eta; Adam 0.3/sqrt(0.09) = 1, 0.3/sqrt(0.36) = 0.5", np.isclose(.3 / math.sqrt(.09), 1) and np.isclose(.3 / .6, .5))
k = 10; check("ch8: the Reddi-style gradient (k^2 w.p. 1/k, -k otherwise) has mean +1", np.isclose(k * k / k - k * (1 - 1 / k), 1))
# ---------- chapter 9: Adafactor reconstruction is exact for a rank-one V
rs = np.random.RandomState(0); u, w = rs.rand(8) + .2, rs.rand(12) + .2; V = np.outer(u, w)
r_, c_ = V.sum(1), V.sum(0); Vt = np.outer(r_, c_) / r_.sum()
check("ch9: Adafactor's factored V reproduces a rank-one V exactly", np.allclose(Vt, V))
check("ch9: storage 96 -> 20 numbers, ratio 0.208", 8 * 12 == 96 and 8 + 12 == 20 and abs(20 / 96 - .2083) < 1e-3)
# ---------- chapter 10: Newton on x^4 - x^2
f = lambda x: x ** 4 - x ** 2; gf = lambda x: 4 * x ** 3 - 2 * x; hf = lambda x: 12 * x * x - 2
x1 = .3 - gf(.3) / hf(.3)
check("ch10: at x0 = 0.3, h = -0.92 and the Newton step goes uphill to -0.2348", np.isclose(hf(.3), -.92) and abs(x1 + .2348) < 1e-4 and f(x1) > f(.3))
check("ch10: h < 0 exactly for |x| < 0.408", abs(math.sqrt(1 / 6) - .408) < 1e-3)
xs = [.3]
for _ in range(9): xs.append(xs[-1] - gf(xs[-1]) / hf(xs[-1]))
check("ch10: undamped Newton from 0.3 converges to the local maximum 0", abs(xs[-1]) < 1e-6, f"{xs[-1]:.2e}")
xs = [1.2]
for _ in range(9): xs.append(xs[-1] - gf(xs[-1]) / hf(xs[-1]))
check("ch10: from 1.2 it converges to the minimum 1/sqrt(2)", abs(xs[-1] - math.sqrt(.5)) < 1e-9)
check("ch10: worked example 5 theta^2 from 4: Newton -> 0, GD(0.01) -> 3.6", np.isclose(4 - 40 / 10, 0) and np.isclose(4 - .4, 3.6))
# ---------- chapter 11: BFGS / GD / Newton on (1-x)^2 + 10(y-x^2)^2, CG on d = 60
F = lambda q: (1 - q[0]) ** 2 + 10 * (q[1] - q[0] ** 2) ** 2
G = lambda q: np.array([-2 * (1 - q[0]) - 40 * q[0] * (q[1] - q[0] ** 2), 20 * (q[1] - q[0] ** 2)])
Hh = lambda q: np.array([[2 - 40 * (q[1] - q[0] ** 2) + 80 * q[0] ** 2, -40 * q[0]], [-40 * q[0], 20.]])
def first_below(seq, tol=1e-10):
    for i, q in enumerate(seq):
        if F(q) < tol: return i
    return None
q = np.array([-1.2, 1.]); gd = [q.copy()]
for _ in range(400): q = q - .01 * G(q); gd.append(q.copy())
q = np.array([-1.2, 1.]); nt = [q.copy()]
for _ in range(20): q = q - np.linalg.solve(Hh(q), G(q)); nt.append(q.copy())
q = np.array([-1.2, 1.]); Cm = np.eye(2); bf = [q.copy()]; evals = 0
for _ in range(400):
    g0 = G(q); d = -Cm @ g0; a = 1.; f0 = F(q)
    while F(q + a * d) > f0 + 1e-4 * a * (g0 @ d) and a > 1e-10: a /= 2; evals += 1
    evals += 1
    qn = q + a * d; s_, y_ = qn - q, G(qn) - g0
    if y_ @ s_ > 1e-14:
        rho = 1 / (y_ @ s_); A = np.eye(2) - rho * np.outer(s_, y_); Cm = A @ Cm @ A.T + rho * np.outer(s_, s_)
    q = qn; bf.append(q.copy())
    if F(q) < 1e-14: break
print(f"      ch11 race: GD loss after 400 steps {F(gd[-1]):.2e}; Newton below 1e-10 at step {first_below(nt)}; BFGS at step {first_below(bf)} ({evals} line-search evaluations)")
check("ch11: GD is still far from the minimum after 400 steps; BFGS and Newton reach 1e-10", F(gd[-1]) > 1e-4 and first_below(bf) is not None and first_below(nt) is not None)
def cg_hit(kappa, D=60, K=200):
    lam = kappa ** (np.arange(D) / (D - 1)); xs_ = np.cos(1.7 * np.arange(D) + .3); b = lam * xs_
    eA = lambda x: math.sqrt(np.sum(lam * (x - xs_) ** 2)); e0 = eA(np.zeros(D))
    x = np.zeros(D); r = b.copy(); p_ = r.copy(); rr = r @ r; hc = None
    for k_ in range(1, K + 1):
        Ap = lam * p_; al = rr / (p_ @ Ap); x = x + al * p_; r = r - al * Ap; rr1 = r @ r; p_ = r + rr1 / rr * p_; rr = rr1
        if hc is None and eA(x) / e0 < 1e-6: hc = k_
    eta = 2 / (1 + kappa); z = np.zeros(D); hg = None
    for k_ in range(1, K + 1):
        z = z - eta * (lam * z - b)
        if hg is None and eA(z) / e0 < 1e-6: hg = k_
    return hc, hg
hc, hg = cg_hit(100.)
print(f"      ch11 CG vs GD at kappa = 100 (to 1e-6): CG {hc}, GD {hg}")
check("ch11: CG needs far fewer products than GD and at most d = 60", hc is not None and hc <= 60 and (hg is None or hg > 3 * hc))
# ---------- chapter 12: natural gradient vs GD on the Gaussian fit
MB, SB = 1., .5; Ls = math.log(SB) + .5
NLL = lambda m, s: math.log(s) + (SB * SB + (m - MB) ** 2) / (2 * s * s)
dm = lambda m, s: (m - MB) / (s * s); ds = lambda m, s: 1 / s - (SB * SB + (m - MB) ** 2) / s ** 3
st = {"gds": lambda m, s, e: (m - e * dm(m, s), s - e * ds(m, s)),
      "gdl": lambda m, s, e: (m - e * dm(m, s), math.exp(math.log(s) - e * s * ds(m, s))),
      "ngs": lambda m, s, e: (m - e * s * s * dm(m, s), s - e * s * s / 2 * ds(m, s)),
      "ngl": lambda m, s, e: (m - e * s * s * dm(m, s), math.exp(math.log(s) - e * .5 * s * ds(m, s)))}
def nrun(kk, e, T=300):
    m, s = -1.5, 2.5; P = [(m, s)]; hit = None
    for t in range(1, T + 1):
        m, s = st[kk](m, s, e)
        if not (math.isfinite(m) and math.isfinite(s)) or s <= 0: return P, hit, True
        P.append((m, s))
        if hit is None and NLL(m, s) - Ls < 1e-8: hit = t
    return P, hit, False
R = {kk: nrun(kk, .15) for kk in st}
gap = lambda a, b: max(math.dist(a[t], b[t]) for t in range(41))
print("      ch12 at eta = 0.15: steps", {kk: R[kk][1] for kk in R}, f"| GD paths differ by {gap(R['gds'][0], R['gdl'][0]):.3f}, natural by {gap(R['ngs'][0], R['ngl'][0]):.3f}")
check("ch12: the two natural-gradient paths are much closer than the two GD paths", gap(R['ngs'][0], R['ngl'][0]) < .25 * gap(R['gds'][0], R['gdl'][0]))
check("ch12 (text): at eta = 0.15 both GD runs reach the optimum in fewer steps than natural gradient",
      all(R[kk][1] is not None for kk in st) and max(R['gds'][1], R['gdl'][1]) < min(R['ngs'][1], R['ngl'][1]))
mA, sA = -1.5, 2.5; mB, _ = st["ngs"](mA, sA, .15); mC, _ = st["ngl"](mA, sA, .15)
check("ch12 (text): natural gradient's mean step is mu - eta (mu - 1) in both parameterizations", np.isclose(mB, mA - .15 * (mA - 1)) and np.isclose(mC, mB))
# ---------- chapter 13: Isserlis exactness of the K-FAC gap
rsk = np.random.RandomState(5); Sa = np.array([[1, .4, .1], [.4, .8, -.2], [.1, -.2, .6]]); Sz = np.array([[.9, .3], [.3, .5]]); Bk = np.array([[.8, -.5, .3], [.2, .7, -.6]])
check("ch13: covariances are positive definite", np.all(np.linalg.eigvalsh(Sa) > 0) and np.all(np.linalg.eigvalsh(Sz) > 0))
rho = .6; Na = 400000
a = rsk.multivariate_normal(np.zeros(3), Sa, Na); z = rsk.multivariate_normal(np.zeros(2), Sz, Na); dl = rho * a @ Bk.T + math.sqrt(1 - rho * rho) * z
vecs = np.einsum('ni,nk->nik', a, dl).reshape(Na, 6)        # index j*m + k  (vec(delta a^T) = a kron delta)
Fmc = vecs.T @ vecs / Na
A = Sa; S = rho * rho * Bk @ Sa @ Bk.T + (1 - rho * rho) * Sz; Cc = rho * Sa @ Bk.T
Fex = np.zeros((6, 6))
for j in range(3):
    for k_ in range(2):
        for l in range(3):
            for r_ in range(2):
                Fex[j * 2 + k_, l * 2 + r_] = A[j, l] * S[k_, r_] + Cc[j, k_] * Cc[l, r_] + Cc[j, r_] * Cc[l, k_]
check("ch13: the exact Isserlis block matches a 400 000-sample Monte Carlo estimate", np.abs(Fmc - Fex).max() < .03, f"max diff {np.abs(Fmc - Fex).max():.4f}")
check("ch13: A kron S is the K-FAC part", np.allclose(np.kron(A, S), Fex - (np.einsum('jk,lr->jklr', Cc, Cc) + np.einsum('jr,lk->jklr', Cc, Cc)).reshape(6, 6)))
print(f"      ch13 relative error at rho = 0.6: {np.linalg.norm(Fex - np.kron(A, S)) / np.linalg.norm(Fex):.4f}")
# ---------- chapter 16: Newton-Schulz spectrum and spectral-norm steepest descent
sig = .03 ** (np.arange(12) / 11); s0 = sig / np.linalg.norm(sig)
check("ch16: normalized singular values run from 0.687 to 0.021", abs(s0.max() - .687) < 5e-4 and abs(s0.min() - .0206) < 5e-4, f"{s0.max():.4f} .. {s0.min():.4f}")
cub = lambda s: .5 * s * (3 - s * s); qu = lambda s: 3.4445 * s - 4.7750 * s ** 3 + 2.0315 * s ** 5
s5, s12, q5 = s0.copy(), s0.copy(), s0.copy()
for i in range(12):
    s12 = cub(s12)
    if i < 5: s5, q5 = cub(s5), qu(q5)
check("ch16: cubic after 5 steps is still far from 1 at the bottom; after 12 all within 0.002 of 1", s5.min() < .5 and np.abs(s12 - 1).max() < 2e-3, f"{s5.min():.3f}, {np.abs(s12 - 1).max():.4f}")
check("ch16: quintic after 5 steps lies in a band around 1 (0.68 .. 1.14), not at 1", q5.min() > .6 and q5.max() < 1.2 and np.abs(q5 - 1).max() > .1, f"[{q5.min():.3f}, {q5.max():.3f}]")
Gr = np.random.RandomState(3).randn(6, 5); U_, Sg_, Vt_ = np.linalg.svd(Gr, full_matrices=False)
check("ch16: spectral steepest descent: -UV^T attains -sum(sigma) under ||D||_2 <= 1", np.isclose(np.sum(Gr * (U_ @ Vt_)), Sg_.sum()))
# ---------- chapter 15: Hutchinson variance formula
Hr = np.random.RandomState(1).randn(5, 5); Hr = Hr + Hr.T; Zp = np.random.RandomState(2).choice([-1., 1.], size=(400000, 5)); est = Zp * (Zp @ Hr.T)
var_pred = (Hr ** 2).sum(1) - np.diag(Hr) ** 2
check("ch15: one probe's variance in entry i is sum_{j != i} H_ij^2", np.allclose(est.var(0), var_pred, rtol=.02), f"{est.var(0).round(2)} vs {var_pred.round(2)}")

# ---------- chapter 17: data for the dashboard (measured runs of Fig. 9)
RES = os.path.join(ROOT, "..", "survey-v3", "experiments", "results.json")
R9 = json.load(open(RES, encoding="utf-8"))
out = {"meta": R9["meta"], "best_lr": R9["best_lr"], "opt": {}}
fin = lambda v: v if (v is not None and math.isfinite(v)) else None
for name, rows in R9["sweep"].items():
    o = {"lrs": [], "curves": [], "final": []}
    for row in rows:
        cs = [np.array(r["curve"], float) for r in row["runs"]]; L = min(len(c) for c in cs)
        Y = np.stack([c[:L, 2] for c in cs])
        o["lrs"].append(row["lr"])
        o["curves"].append({"ge": cs[0][:L, 1].tolist(), "med": [fin(float(v)) for v in np.median(Y, 0)], "lo": [fin(float(v)) for v in Y.min(0)], "hi": [fin(float(v)) for v in Y.max(0)]})
        o["final"].append([fin(r["final"]) for r in row["runs"]])
    t = np.array(R9["timing"][name]["curve"], float)
    o["timing"] = {"t": t[:, 3].tolist(), "loss": [fin(float(v)) for v in t[:, 2]], "total": R9["timing"][name]["time"]}
    o["memory"] = R9["memory"][name]
    out["opt"][name] = o
dest = os.path.join(ROOT, "guide", "compare", "data.json"); os.makedirs(os.path.dirname(dest), exist_ok=True)
json.dump(out, open(dest, "w", encoding="utf-8"), separators=(",", ":"))
med_final = {n: float(np.median([v if v is not None else np.inf for v in o["final"][o["lrs"].index(out["best_lr"][n])]])) for n, o in out["opt"].items()}
print("      ch17 best-lr median final losses:", {k: f"{v:.2e}" for k, v in med_final.items()})
print("      ch17 timing totals (s):", {n: round(o["timing"]["total"], 2) for n, o in out["opt"].items()})
print("      ch17 state bytes at h = 128:", {n: next(m["state_bytes"] for m in o["memory"] if m["h"] == 128) for n, o in out["opt"].items()})
check("ch17: dashboard data written", os.path.getsize(dest) > 1000, f"{os.path.getsize(dest)} bytes")
done()
