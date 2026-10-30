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
done()
