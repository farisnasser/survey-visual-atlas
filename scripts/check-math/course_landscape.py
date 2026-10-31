"""Crash course stop 11 (course/landscape): symmetry, benign landscapes, a spurious minimum, the PL condition, and
GD on wide networks. Every number on the page is recomputed here. RNG = a port of src/lib/math.js rng (mulberry32 +
Box-Muller), so the visuals' random starts and initial weights are the same as here."""
import math
from collections import Counter
from common import *
np.seterr(over="ignore")

print("course_landscape")


def rng(seed):
    a = [seed & 0xffffffff]; spare = [None]
    imul = lambda x, y: ((x & 0xffffffff) * (y & 0xffffffff)) & 0xffffffff

    def nxt():
        a[0] = (a[0] + 0x6d2b79f5) & 0xffffffff; t = a[0]
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296

    def normal():
        if spare[0] is not None:
            s = spare[0]; spare[0] = None; return s
        u = 0
        while u == 0: u = nxt()
        v = nxt(); r = math.sqrt(-2 * math.log(u)); spare[0] = r * math.sin(2 * math.pi * v)
        return r * math.cos(2 * math.pi * v)
    nxt.normal = normal
    return nxt


def classify(g, starts, eta, T, pts, tol=1e-3):
    out = []
    for s in starts:
        p = gd(g, s, eta, T)[-1]
        out.append(next((k for k, q in pts.items() if np.linalg.norm(p - q) < tol), "other"))
    return Counter(out)


# ---------- 1. symmetry: two hidden tanh units, mirror-image minima (visual 'mirror') ----------
XS = np.linspace(-2, 2, 9); TA, TB = 1.6, .4; YS = np.tanh(TA * XS) + np.tanh(TB * XS)
mL = lambda p: np.mean((np.tanh(p[0] * XS) + np.tanh(p[1] * XS) - YS) ** 2)
mG = lambda p, np=np: (lambda r: np.array([np.mean(2 * r * XS / np.cosh(p[0] * XS) ** 2), np.mean(2 * r * XS / np.cosh(p[1] * XS) ** 2)]))(np.tanh(p[0] * XS) + np.tanh(p[1] * XS) - YS)
check_grad("mirror loss", mL, mG, rand_pts(20, -.5, 2.5))
check("mirror: L(w1, w2) = L(w2, w1) (swapping the units)", all(np.isclose(mL(p), mL(p[::-1])) for p in rand_pts(10, -1, 3)))
crit = []
for s in rand_pts(200, -.5, 2.5, seed=4):
    p = s.copy()
    for _ in range(60):
        try: p = p - np.linalg.solve(fd_hess(mL, p), mG(p))
        except np.linalg.LinAlgError: break
    if np.all(np.isfinite(p)) and np.linalg.norm(mG(p)) < 1e-9 and np.abs(p - 1).max() < 1.6 and not any(np.linalg.norm(p - c) < 1e-5 for c in crit):
        crit.append(p)
crit = sorted(crit, key=lambda c: (round(c[0], 3), round(c[1], 3)))
info = [(np.round(c, 3).tolist(), round(mL(c), 4), np.round(eig(fd_hess(mL, c)), 3).tolist()) for c in crit]
print("  mirror critical points in [-0.5, 2.5]^2:", info)
check("mirror: exactly three critical points: minima (0.4, 1.6), (1.6, 0.4) with loss 0, saddle (0.736, 0.736) on the diagonal",
      len(crit) == 3 and np.allclose(crit[0], [.4, 1.6], atol=1e-6) and np.allclose(crit[2], [1.6, .4], atol=1e-6) and np.allclose(crit[1], [.7363, .7363], atol=1e-4))
SAD = crit[1]
ev = eig(fd_hess(mL, SAD))
check("mirror: saddle loss 0.0118, Hessian eigenvalues -0.106 (across the diagonal) and 0.775 (along it)", round(mL(SAD), 4) == .0118 and round(ev[0], 3) == -.106 and round(ev[1], 3) == .775)
check("mirror: minima Hessian eigenvalues 0.031 and 1.505", np.allclose(np.round(eig(fd_hess(mL, crit[0])), 3), [.031, 1.505]))
p = np.array([2., 2.])
for _ in range(500): p = p - .5 * mG(p); assert p[0] == p[1]
check("mirror: GD started on the diagonal stays exactly on it (the units stay identical)", True)
R3 = rng(3); MSTART = [[-.3 + 2.8 * R3(), -.3 + 2.8 * R3()] for _ in range(24)]
cnt = classify(mG, MSTART, .5, 3000, {"a": crit[2], "b": crit[0]})
check("mirror visual: 24 random starts, GD step 0.5, 3000 steps: 12 end at (1.6, 0.4), 12 at (0.4, 1.6)", cnt == Counter({"a": 12, "b": 12}), f"{cnt}")
ext = max(np.abs(gd(mG, s, .5, 3000) - 1).max() for s in MSTART + [[2 + d, 2 - d] for d in (-.1, -.01, 0, .01, .1)])
check("mirror visual: every path stays inside the drawn square [-0.5, 2.5]^2", ext < 1.5, f"max |p - 1| = {ext:.3f}")
for d, want in ((0., "s"), (1e-3, "a"), (-1e-3, "b"), (.05, "a"), (-.1, "b")):
    end = gd(mG, [2 + d, 2 - d], .5, 3000)[-1]
    got = "s" if np.linalg.norm(end - SAD) < 1e-3 else "a" if np.linalg.norm(end - crit[2]) < 1e-3 else "b" if np.linalg.norm(end - crit[0]) < 1e-3 else "?"
    assert got == want, (d, end)
check("mirror visual: start (2 + d, 2 - d): d = 0 ends at the saddle; d > 0 at (1.6, 0.4); d < 0 at (0.4, 1.6) (d = +-0.001, 0.05, -0.1)", True)

# ---------- 2. benign: f(u) = 1/2 ||u u^T - diag(1, s)||_F^2 (visual 'benign') ----------
def bL(u, s=.5): return .5 * np.sum((np.outer(u, u) - np.diag([1, s])) ** 2)
def bG(u, s=.5): return 2 * (np.outer(u, u) - np.diag([1, s])) @ u
check_grad("benign", bL, bG, rand_pts(20, -1.5, 1.5))
check("benign: at 0 the Hessian is -2 diag(1, s): a local maximum", np.allclose(fd_hess(bL, [0, 0]), -2 * np.diag([1, .5]), atol=1e-6))
check("benign: at (+-1, 0) loss 0.125 = s^2/2, Hessian eigenvalues 1 and 4 (minimum, global)",
      np.isclose(bL([1, 0]), .125) and np.allclose(eig(fd_hess(bL, [1, 0])), [1, 4], atol=1e-5) and np.allclose(eig(fd_hess(bL, [-1, 0])), [1, 4], atol=1e-5))
check("benign: global min value: f >= s^2/2 everywhere on a grid", min(bL(p) for p in rand_pts(4000, -2, 2, seed=1)) >= .125 - 1e-12)
check("benign: at (0, +-sqrt(s)) loss 0.5, Hessian eigenvalues -1 and 2 (strict saddle)",
      np.isclose(bL([0, np.sqrt(.5)]), .5) and np.allclose(eig(fd_hess(bL, [0, np.sqrt(.5)])), [-1, 2], atol=1e-5))
check("benign: general s: saddle eigenvalues 2(s - 1) and 4s, minimum eigenvalues 2(1 - s) and 4",
      all(np.allclose(eig(fd_hess(lambda u: bL(u, s), [0, np.sqrt(s)])), sorted([2 * (s - 1), 4 * s]), atol=1e-5) and np.allclose(eig(fd_hess(lambda u: bL(u, s), [1, 0])), sorted([2 * (1 - s), 4]), atol=1e-5) for s in (.2, .5, .8)))
R5 = rng(5); BSTART = [[-1.4 + 2.8 * R5(), -1.4 + 2.8 * R5()] for _ in range(24)]
ok = True
for s in np.round(np.arange(.1, .9001, .05), 2):
    c = classify(lambda u: bG(u, s), BSTART, .1, 2000, {"+": np.array([1, 0]), "-": np.array([-1, 0])})
    ok &= c["+"] + c["-"] == 24
    e = gd(lambda u: bG(u, s), [0, 1.2], .1, 2000)[-1]; ok &= bool(np.allclose(e, [0, np.sqrt(s)], atol=1e-6))
    ok &= max(np.abs(gd(lambda u: bG(u, s), x, .1, 2000)).max() for x in BSTART + [[0, 1.2]]) < 1.5
check("benign visual: for every s on the slider (0.1 ... 0.9): all 24 random starts reach a global minimum (+-1, 0); the start (0, 1.2) on the saddle's stable set ends at the saddle (0, sqrt s); paths stay in [-1.5, 1.5]^2", ok)
c5 = classify(bG, BSTART, .1, 2000, {"+": np.array([1, 0]), "-": np.array([-1, 0])})
check("benign visual at s = 0.5: 13 end at (1, 0), 11 at (-1, 0)", c5 == Counter({"+": 13, "-": 11}), f"{c5}")

# ---------- 3. a spurious minimum: one neuron, one example, F(v, w) = (1 - v phi(w))^2 (visual 'neuron') ----------
ACT = {"quad": (lambda z: -(z - 1) ** 2 / 8, lambda z: -(z - 1) / 4), "tanh": (np.tanh, lambda z: 1 / np.cosh(z) ** 2)}
for k, (ph, dph) in ACT.items():
    check_grad(f"neuron, phi = {k}", lambda p: (1 - p[0] * ph(p[1])) ** 2, lambda p: np.array([-2 * (1 - p[0] * ph(p[1])) * ph(p[1]), -2 * (1 - p[0] * ph(p[1])) * p[0] * dph(p[1])]), rand_pts(10, -3, 3))
ph, dph = ACT["quad"]; Fq = lambda p: (1 - p[0] * ph(p[1])) ** 2
ring = min(Fq(np.array([1.5, 1]) + e * np.array([np.cos(t), np.sin(t)])) for e in np.linspace(1e-3, .3, 40) for t in np.linspace(0, 2 * np.pi, 361))
check("neuron, quadratic phi: (1.5, 1) is a local minimum with F = 1 (F >= 1 within radius 0.3), the global min is 0", ring >= 1 - 1e-12 and Fq([1.5, 1]) == 1 and Fq([-8, 2]) == 0)
check("neuron, quadratic phi: Hessian at (v, 1), v > 0, has eigenvalues 0 and v/2 (flat along v)", np.allclose(eig(fd_hess(Fq, [1.5, 1])), [0, 1.5 / 2], atol=1e-5))


def neuron_runs(k, starts, T=6000, eta=.05):
    ph, dph = ACT[k]; P = np.array(starts, float)
    with np.errstate(all="ignore"):
        for _ in range(T):
            e = 1 - P[:, 0] * ph(P[:, 1]); P = P - eta * np.stack([-2 * e * ph(P[:, 1]), -2 * e * P[:, 0] * dph(P[:, 1])], 1)
    F = (1 - P[:, 0] * ph(P[:, 1])) ** 2
    return int((F < 1e-6).sum()), int((np.abs(F - 1) < 1e-3).sum()), P


R9 = rng(9); NSTART = [[-3 + 6 * R9(), -3 + 6 * R9()] for _ in range(30)]
NQ, NT = neuron_runs("quad", NSTART), neuron_runs("tanh", NSTART)
print(f"  neuron visual (30 starts): quad {NQ[:2]}, tanh {NT[:2]}; max |p| {np.abs(NQ[2]).max():.2f} / {np.abs(NT[2]).max():.2f}")
check("neuron visual: 30 random starts, step 0.05, 6000 steps: quadratic phi 21 reach F = 0 and 9 stop at F = 1; tanh 30 reach F = 0",
      NQ[:2] == (21, 9) and NT[:2] == (30, 0))
G41 = np.linspace(-3, 3, 41); GV, GW = np.meshgrid(G41, G41); GRID = np.c_[GV.ravel(), GW.ravel()]
gq, gt = neuron_runs("quad", GRID), neuron_runs("tanh", GRID)
check("neuron: 41 x 41 grid of starts: quadratic phi 30.4% stop at F = 1; tanh: only the start (0, 0) (a saddle)",
      round(100 * gq[1] / 1681, 1) == 30.4 and gt[1] == 1)

# ---------- 4. the PL condition: f(x) = x^2 + 3 sin^2 x (visual 'pl') ----------
xx = np.linspace(-20, 20, 4000001); xx = xx[np.abs(xx) > 1e-6]
fP = lambda x: x * x + 3 * np.sin(x) ** 2; dP = lambda x: 2 * x + 3 * np.sin(2 * x)
ratio = .5 * dP(xx) ** 2 / fP(xx); MU = ratio.min(); XMU = abs(xx[np.argmin(ratio)])
check("PL: mu = min (1/2) f'^2 / (f - f*) = 0.1755 at |x| = 2.20 (f* = 0 at x = 0)", round(MU, 4) == .1755 and round(XMU, 2) == 2.2)
check("PL: near 0 the ratio tends to 8; as |x| grows it tends to 2", abs(ratio[np.abs(xx) < 1e-3].mean() - 8) < 1e-3 and abs(.5 * dP(1e4) ** 2 / fP(1e4) - 2) < 1e-3)
d2 = 2 + 6 * np.cos(2 * xx)
check("PL: f'' ranges over [-4, 8]: non-convex, L = 8", np.isclose(d2.min(), -4, atol=1e-6) and np.isclose(d2.max(), 8, atol=1e-6))
check("PL: x = 0 is the only critical point (|f'| >= 1.54 for |x| > 0.5)", np.abs(dP(xx[np.abs(xx) > .5])).min() > 1.54)
for eta in np.round(np.arange(.01, .2401, .005), 3):
    for x0 in np.round(np.arange(-6, 6.01, .5), 1):
        x, f0, ok = x0, fP(x0), True
        q = 1 - 2 * MU * eta * (1 - 8 * eta / 2)
        for t in range(1, 301):
            x -= eta * dP(x); ok &= fP(x) <= q ** t * f0 + 1e-14
        assert ok, (eta, x0)
check("PL: GD obeys f(x_t) <= (1 - 2 mu eta (1 - L eta/2))^t f(x_0) for every slider eta in [0.01, 0.24] and starts -6 ... 6", True)
check("PL: with eta = 1/L the factor is 1 - mu/L = 0.978", round(1 - MU / 8, 3) == .978)
x, steps = 5.5, 0
while fP(x) > 1e-10: x -= dP(x) / 8; steps += 1
print(f"  PL: from x0 = 5.5 with eta = 1/8, f < 1e-10 after {steps} steps; bound needs {math.ceil(math.log(1e-10 / fP(5.5)) / math.log(1 - MU / 8))}")
check("PL text: from x0 = 5.5 (f = 31.7), eta = 1/8: f < 1e-10 after 11 steps; the bound promises it by step 1194",
      round(fP(5.5), 1) == 31.7 and steps == 11 and math.ceil(math.log(1e-10 / fP(5.5)) / math.log(1 - MU / 8)) == 1194)

# ---------- 5. wide networks: f(x) = m^{-1/2} sum_r a_r tanh(w_r . x), train W by GD (visual 'wide') ----------
NW = 8
TT = np.array([2 * math.pi * i / NW + .3 for i in range(NW)])
XW = np.stack([np.cos(TT), np.sin(TT), np.ones(NW)], 1) / math.sqrt(2); YW = np.sin(2 * TT) + .5 * np.cos(3 * TT)


def wide(m, T=1000, seed=7):
    R = rng(seed); W = np.array([[R.normal() for _ in range(3)] for _ in range(m)]); a = np.array([1. if r % 2 == 0 else -1. for r in range(m)]); W0 = W.copy()
    gram = lambda W: (lambda S: (S @ S.T) / m * (XW @ XW.T))(1 / np.cosh(XW @ W.T) ** 2)
    e0 = np.linalg.eigvalsh(gram(W)); eta = 1 / e0[-1]; losses = []
    for k in range(T + 1):
        Z = XW @ W.T; r = np.tanh(Z) @ a / math.sqrt(m) - YW; losses.append(.5 * r @ r)
        if k == T: break
        W = W - eta * ((r[:, None] * (1 / np.cosh(Z) ** 2) * a[None, :]).T @ XW) / math.sqrt(m)
    eT = np.linalg.eigvalsh(gram(W))
    return dict(lam0=e0[0], lmax=e0[-1], eta=eta, L=np.array(losses), move=np.linalg.norm(W - W0) / np.linalg.norm(W0), dlam=abs(eT[0] - e0[0]) / e0[0])


# check the backprop gradient once
m_, R_ = 6, rng(1); W_ = np.array([[R_.normal() for _ in range(3)] for _ in range(m_)]); a_ = np.array([1., -1, 1, -1, 1, -1])
lw = lambda v: (lambda r: .5 * r @ r)(np.tanh(XW @ v.reshape(m_, 3).T) @ a_ / math.sqrt(m_) - YW)
gw = lambda v: (lambda Z: ((((np.tanh(Z) @ a_ / math.sqrt(m_) - YW)[:, None]) * (1 / np.cosh(Z) ** 2) * a_[None, :]).T @ XW / math.sqrt(m_)).ravel())(XW @ v.reshape(m_, 3).T)
check_grad("wide net (first-layer weights)", lw, gw, [W_.ravel(), W_.ravel() * .5])
RES = {m: wide(m) for m in (4, 16, 64, 256, 1024)}
for m, r_ in RES.items():
    print(f"  m = {m:5d}: lam_min(G0) = {r_['lam0']:.4f}, lam_max = {r_['lmax']:.3f}, L(0) = {r_['L'][0]:.3f}, L(1000) = {r_['L'][-1]:.2e}, move = {r_['move']:.3f}, d lam_min = {100 * r_['dlam']:.1f}%")
check("wide: relative weight movement 4.09 / 1.59 / 0.838 / 0.374 / 0.174 for m = 4 ... 1024 (about halves per 4x width)",
      [round(RES[m]["move"], 3 if RES[m]["move"] < 1 else 2) for m in RES] == [4.09, 1.59, .838, .374, .174])
check("wide: change of lambda_min(G) during training 96.5% / 37.5% / 95.9% / 18.6% / 0.9%",
      [round(100 * RES[m]["dlam"], 1) for m in RES] == [96.5, 37.5, 95.9, 18.6, .9])
check("wide: m = 4 is still at loss 0.064 after 1000 steps; m >= 16 below 1e-6", round(RES[4]["L"][-1], 3) == .064 and all(RES[m]["L"][-1] < 1e-6 for m in (16, 64, 256, 1024)))
check("wide: at m = 1024: lambda_min(G0) = 0.0085, eta = 1/lambda_max = 0.605", round(RES[1024]["lam0"], 4) == .0085 and round(RES[1024]["eta"], 3) == .605)
check("wide: every run stays below Du et al.'s rate (1 - eta lambda0/2)^k L(0)",
      all(np.all(r_["L"] <= (1 - r_["eta"] * r_["lam0"] / 2) ** np.arange(1001) * r_["L"][0] + 1e-12) for r_ in RES.values()))
check("wide: movement x sqrt(m) is roughly constant for m >= 64 (5.6-6.7)", all(5.5 < RES[m]["move"] * math.sqrt(m) < 6.8 for m in (64, 256, 1024)),
      f"{[round(RES[m]['move'] * math.sqrt(m), 2) for m in RES]}")
done()
