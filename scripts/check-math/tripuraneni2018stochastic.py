"""Tripuraneni et al. 2018: stochastic cubic regularization (Alg. 1) with Cubic-Subsolver (Alg. 3).

m(D) = g^T D + 1/2 D^T B D + (rho/6)|D|^3 with sampled g (S1 gradients) and B (S2 Hessians).
Subsolver: if |g| >= ell^2/rho: Cauchy step D = -R_c g/|g|, R_c = -g^T B g/(rho|g|^2) + sqrt((g^T B g/(rho|g|^2))^2 + 2|g|/rho);
else GD on m with g~ = g + sigma*zeta and step 1/(20 ell).
"""
from common import *

print("tripuraneni2018stochastic")
# notes' 1D example: f = cos s at x = 0, sampled g = 0.01, B = -0.9, rho = 1
g1, B1, rho = .01, -.9, 1.0
m1 = lambda D: g1 * D + .5 * B1 * D * D + rho / 6 * abs(D) ** 3
Ds = np.linspace(-3, 3, 600001); Dst = Ds[np.argmin(m1(Ds))]
q1 = lambda D: -.5 * D * D + abs(D) ** 3 / 6          # model with the true g = 0, H = -1
check("notes: |D*| ~ 1.811043", abs(abs(Dst) - 1.811043) < 2e-5, f"D* = {Dst:.6f}")
check("notes: m(D*) ~ -0.504055, true model ~ -0.649939, true change cos(D*) - 1 ~ -1.237943",
      abs(m1(Dst) + .504055) < 1e-5 and abs(q1(Dst) + .649939) < 1e-5 and abs(np.cos(Dst) - 1 + 1.237943) < 1e-5)


def model(g, B, rho):
    return (lambda D: g @ D + .5 * D @ B @ D + rho / 6 * np.linalg.norm(D) ** 3), (lambda D: g + B @ D + rho / 2 * np.linalg.norm(D) * D)


def global_min(g, B, rho):
    """Exact global minimizer of the cubic model (secular equation in r = |D|)."""
    lam, V = np.linalg.eigh(B); gb = V.T @ g
    lo = max(0, -2 * lam[0] / rho) + 1e-14
    nr = lambda r: np.linalg.norm(gb / (lam + rho * r / 2))
    if nr(lo) < lo:
        Db = np.where(lam + rho * lo / 2 > 1e-12, -gb / (lam + rho * lo / 2), 0); Db[0] = np.sqrt(max(lo ** 2 - Db @ Db, 0))
        return V @ Db
    hi = lo + 1
    while nr(hi) > hi:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if nr(mid) > mid else (lo, mid)
    return V @ (-gb / (lam + rho * hi / 2))


R = np.random.default_rng(0)
for trial in range(6):
    g = R.normal(size=2) * (.3 if trial < 3 else 2); A = R.normal(size=(2, 2)); B = (A + A.T) / 2
    mf, mg = model(g, B, rho)
    check_grad(f"model {trial}", mf, mg, rand_pts(5, -2, 2, seed=trial))
    gn = np.linalg.norm(g); k = g @ B @ g / (rho * gn ** 2)
    Rc = -k + np.sqrt(k * k + 2 * gn / rho)
    ts = np.linspace(0, 6, 60001); best = ts[np.argmin([mf(-t * g / gn) for t in ts])]
    check(f"Cauchy radius R_c minimizes m along -g (trial {trial})", abs(best - Rc) < 2e-4, f"R_c = {Rc:.4f}, scan {best:.4f}")
    # Alg. 3 subsolver
    ell = np.max(np.abs(np.linalg.eigvalsh(B))) + rho * 2
    if gn >= ell ** 2 / rho:
        D = -Rc * g / gn
    else:
        gt = g + 1e-3 * R.normal(size=2); D = np.zeros(2); mt = model(gt, B, rho)[1]
        for _ in range(20000):
            D = D - mt(D) / (20 * ell)
    Dg = global_min(g, B, rho)
    print(f"  trial {trial}: subsolver m = {mf(D):.5f}, global m = {mf(Dg):.5f}, branch {'Cauchy' if gn >= ell ** 2 / rho else 'GD'}")
    if gn < ell ** 2 / rho:
        check(f"perturbed GD on the model reaches its global minimum up to the 1e-3 perturbation (trial {trial})", mf(D) - mf(Dg) < 5e-5, f"gap {mf(D) - mf(Dg):.1e}")

# outer loop on f = x^2/2 + cos y (saddle line y = 0, minima y = +-pi), noisy oracles
f = lambda p: .5 * p[0] ** 2 + np.cos(p[1])
gf = lambda p: np.array([p[0], -np.sin(p[1])])
Hf = lambda p: np.diag([1, -np.cos(p[1])])
check_grad("f", f, gf, rand_pts(20, -3, 3))
eps, ell, sg, sh = .05, 1.0, .5, .5


def scr(S1, S2, seed, T=200):
    Rr = np.random.default_rng(seed); x = np.array([1.2, 0.0]); path = [x.copy()]
    for t in range(T):
        g = gf(x) + sg / np.sqrt(S1) * Rr.normal(size=2)
        Z = Rr.normal(size=(2, 2)); B = Hf(x) + sh / np.sqrt(S2) * (Z + Z.T) / 2
        mf, mg = model(g, B, rho); gn = np.linalg.norm(g)
        if gn >= ell ** 2 / rho:
            k = g @ B @ g / (rho * gn ** 2); D = -(-k + np.sqrt(k * k + 2 * gn / rho)) * g / gn
        else:
            gt = g + 1e-3 * Rr.normal(size=2); mt = model(gt, B, rho)[1]; D = np.zeros(2)
            for _ in range(300):
                D = D - mt(D) / (20 * (ell + 2 * rho))
        if mf(D) >= -np.sqrt(eps ** 3 / rho) / 100:
            return x, t, np.array(path)
        x = x + D; path.append(x.copy())
    return x, T, np.array(path)


ends = [scr(20, 20, s) for s in range(20)]
check("SCR escapes the saddle line and stops near a minimum y = +-pi", all(abs(abs(e[0][1]) - np.pi) < .25 and abs(e[0][0]) < .2 for e in ends), f"median outer steps {np.median([e[1] for e in ends])}")

# in-page notes (papers/tripuraneni2018stochastic/notes.md): notes Sec. 3 cosine example in full, primer numbers
r_ = .9 + np.sqrt(.83)
m_ = lambda s_: .01 * s_ - .45 * s_ ** 2 + abs(s_) ** 3 / 6
q_ = lambda s_: -s_ ** 2 / 2 + abs(s_) ** 3 / 6
check("notes Sec. 3: r solves r^2/2 - 0.9 r - 0.01 = 0, r ~ 1.811043; m(-r) ~ -0.504055", abs(r_ ** 2 / 2 - .9 * r_ - .01) < 1e-12 and abs(r_ - 1.811043) < 1e-6 and abs(m_(-r_) + .504055) < 1e-6)
sg_ = np.linspace(-4, 4, 80001)
check("notes Sec. 3: m is minimized at s = -r (global, on a grid)", abs(sg_[np.argmin(m_(sg_))] + r_) < 2e-4)
check("notes Sec. 3: q(-r) ~ -0.649939, cos r - 1 ~ -1.237943; q minimized at +-2 with q = -2/3", abs(q_(-r_) + .649939) < 1e-6 and abs(np.cos(r_) - 1 + 1.237943) < 1e-6 and abs(sg_[np.argmin(q_(sg_))] ** 2 - 4) < 1e-3 and np.isclose(q_(2.), -2 / 3))
check("notes Sec. 3: bound 0.01 r + 0.05 r^2 ~ 0.182104, exact |m - q| ~ 0.145883", abs(.01 * r_ + .05 * r_ ** 2 - .182104) < 1e-6 and abs(abs(m_(-r_) - q_(-r_)) - .145883) < 1e-6)
Fz = lambda s_, zh, zg: (1 + zh) * np.cos(s_) + zg * np.sin(s_)
gz = [(Fz(1e-6, zh, zg) - Fz(-1e-6, zh, zg)) / 2e-6 for zh in (-.1, .1) for zg in (-.01, .01)]
hz = [(Fz(1e-4, zh, zg) - 2 * Fz(0, zh, zg) + Fz(-1e-4, zh, zg)) / 1e-8 for zh in (-.1, .1) for zg in (-.01, .01)]
check("notes Sec. 3: F(s; zeta) components are unbiased (mean grad 0, mean Hessian -1) and can give g = 0.01, B = -0.9", abs(np.mean(gz)) < 1e-8 and abs(np.mean(hz) + 1) < 1e-4 and any(abs(a - .01) < 1e-8 and abs(b + .9) < 1e-4 for a, b in zip(gz, hz)))
e_, rho_ = 1e-4, 2.
check("notes Sec. 3: eps * sqrt(eps/rho) = sqrt(rho eps) * (eps/rho) = eps^1.5/sqrt(rho)", np.isclose(e_ * np.sqrt(e_ / rho_), e_ ** 1.5 / np.sqrt(rho_)) and np.isclose(np.sqrt(rho_ * e_) * e_ / rho_, e_ ** 1.5 / np.sqrt(rho_)))
check("notes Sec. 5: eps^-1.5 * eps^-2 = eps^-3.5 and eps^-1.5 * eps^-1 * eps^-0.5 = eps^-3", np.isclose(e_ ** -1.5 * e_ ** -2, e_ ** -3.5) and np.isclose(e_ ** -1.5 * e_ ** -1 * e_ ** -.5, e_ ** -3))
ss_ = np.linspace(-6, 6, 12001)
check("primer: cos s - 1 <= -s^2/2 + |s|^3/6 for all s (rho = 1 upper bound); at s = 2 true change ~ -1.416", np.all(np.cos(ss_) - 1 <= q_(ss_) + 1e-12) and abs(np.cos(2) - 1 + 1.416) < 5e-4)
gf = lambda p_: np.array([2 * p_[0] * p_[1], p_[0] ** 2 + 3 * p_[1] ** 2])
hv = (gf(np.array([1 + 1e-6, 2.])) - gf(np.array([1., 2.]))) / 1e-6
check("primer: f = x^2 y + y^3 at (1,2), v = (1,0): Hv = (4, 2); difference quotient matches to ~6 digits", np.allclose(np.array([[4, 2], [2, 12]]) @ [1, 0], [4, 2]) and np.allclose(hv, [4, 2], atol=1e-5))
check("primer: sigma = rho = 1, eps = 0.01: n1 ~ 1/eps^2 = 1e4, n2 ~ 1/(rho eps) = 100", np.isclose(1 / .01 ** 2, 1e4) and np.isclose(1 / .01, 100))
done()
