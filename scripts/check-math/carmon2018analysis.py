"""Carmon & Duchi 2018: Krylov subspace (Lanczos) solutions of the trust-region problem; Theorem 1 bounds; hard case.

f(x) = 1/2 x^T A x + b^T x,  ||x|| <= R.  s_t = argmin over K_t(A, b) = span{b, Ab, ..., A^{t-1} b}.
"""
from common import *

print("carmon2018analysis")


def tr_solve(evals, beta, R):
    """Global TR solution in an eigenbasis: A = diag(evals), linear term beta. Returns (x, lam)."""
    lmin = evals.min()
    if lmin > 0:
        x = -beta / evals
        if np.linalg.norm(x) <= R:
            return x, 0.0
    lo = max(0, -lmin) + 1e-15
    nrm = lambda lam: np.linalg.norm(beta / (evals + lam))
    if nrm(lo) < R:  # hard case: add a u_min component
        x = np.where(evals + lo > 1e-12, -beta / (evals + lo), 0.0)
        i = int(np.argmin(evals)); x[i] = np.sqrt(max(R * R - (x ** 2).sum(), 0))
        return x, -lmin
    hi = lo + 1
    while nrm(hi) > R:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if nrm(mid) > R else (lo, mid)
    lam = (lo + hi) / 2
    return -beta / (evals + lam), lam


def lanczos_tr(A, b, R, T):
    """Krylov TR values f(s_t), t = 1..T (Lanczos with full reorthogonalization)."""
    d = len(b); Q = np.zeros((d, T)); al = []; be = []
    q = b / np.linalg.norm(b); vals = []
    for t in range(T):
        Q[:, t] = q
        w = A @ q
        a = q @ w; al.append(a)
        w = w - Q[:, :t + 1] @ (Q[:, :t + 1].T @ w)
        w = w - Q[:, :t + 1] @ (Q[:, :t + 1].T @ w)
        Tm = np.diag(al) + np.diag(be, 1) + np.diag(be, -1)
        ev, V = np.linalg.eigh(Tm)
        beta = V.T @ (np.linalg.norm(b) * np.eye(t + 1)[0])
        z, _ = tr_solve(ev, beta, R)
        vals.append(.5 * z @ (ev * z) + beta @ z)
        nb = np.linalg.norm(w)
        if nb < 1e-12:
            vals += [vals[-1]] * (T - t - 1); break
        be.append(nb); q = w / nb
    return np.array(vals)


# notes' 2D hard case
A2 = np.diag([-2., 1.]); b2 = np.array([0., 1.])
x, lam = tr_solve(np.diag(A2), b2, 1.0)
fv = .5 * x @ A2 @ x + b2 @ x
check("notes: global TR value -7/6 at (+-2 sqrt2/3, -1/3)", abs(fv + 7 / 6) < 1e-9 and np.allclose(np.abs(x), [2 * np.sqrt(2) / 3, 1 / 3]), f"x = {np.round(x, 4)}, lam = {lam:.3f}")
k1 = lanczos_tr(A2, b2, 1.0, 2)
check("notes: Krylov space span{b} stalls at -1/2 (hard case)", abs(k1[0] + .5) < 1e-9 and abs(k1[-1] + .5) < 1e-9)
k1p = lanczos_tr(A2, b2 + np.array([1e-3, 0]), 1.0, 2)
check("perturbed b: t = 2 Krylov space is R^2 and reaches ~ -7/6", abs(k1p[-1] + 7 / 6) < 5e-3, f"{k1p[-1]:.5f}")

# a d = 400 instance
rng = np.random.default_rng(0)
d, R, T = 400, 1.0, 120
ev = np.concatenate([[-.6], np.linspace(-.55, 1.0, d - 1)])
A = np.diag(ev)
out = {}
cases = {}
bg = rng.normal(size=d); bg *= .1 / np.linalg.norm(bg)        # small b: lam* close to -lam_min (ill-conditioned)
cases["generic"] = bg
bh = bg.copy(); bh[0] = 0; bh *= .05 / np.linalg.norm(bh)       # u_min^T b = 0 and small b: hard case
cases["hard"] = bh
bp = bh + 1e-4 * rng.normal(size=d)                                # Corollary-3-style perturbation
cases["perturbed"] = bp
for name, b in cases.items():
    xs, lam = tr_solve(ev, b, R)
    fs = .5 * xs @ (ev * xs) + b @ xs
    vals = lanczos_tr(A, b, R, T)
    gap = np.maximum(vals - fs, 1e-16)
    hard = abs(b[0]) < 1e-14 and abs(lam + ev.min()) < 1e-9
    lin = 36 * (0 - fs) * np.exp(-4 * np.arange(1, T + 1) * np.sqrt(max(ev.min() + lam, 0) / (ev.max() + lam)))
    sub = None
    if abs(b[0]) > 0:
        sub = (ev.max() - ev.min()) * (xs @ xs) / (np.arange(1, T + 1) - .5) ** 2 * (4 + (ev.min() < 0) / 8 * np.log(4 * (b @ b) / b[0] ** 2) ** 2)
    print(f"  {name}: lam* = {lam:.4f}, hard case {hard}, f* = {fs:.6f}, gap t=10 {gap[9]:.2e}, t=60 {gap[59]:.2e}, t=120 {gap[-1]:.2e}")
    if sub is not None:
        check(f"{name}: gap below both Theorem 1 bounds for all t", np.all(gap <= lin + 1e-12) and np.all(gap <= sub + 1e-12))
    out[name] = {"gap": sig(gap, 4), "lin": sig(np.minimum(lin, 1e6), 4), "sub": (sig(np.minimum(sub, 1e6), 4) if sub is not None else None), "lam": lam, "fstar": fs, "hard": bool(hard)}
check("hard case: Krylov solutions never reach f* (gap stays > 1e-3)", out["hard"]["hard"] and min(out["hard"]["gap"]) > 1e-3)
check("perturbed hard case converges (gap < 1e-6 by t = 120)", out["perturbed"]["gap"][-1] < 1e-6)
check("generic instance converges", out["generic"]["gap"][-1] < 1e-8)
save_json("carmon2018analysis", "data.json", {"T": T, "d": d, "cases": out})
done()
