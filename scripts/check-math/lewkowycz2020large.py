"""Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gur-Ari 2020: the catapult mechanism.

Warmup model (Sec. 2.1): one example x = 1, label y = 0, f = n^{-1/2} v.u, L = f^2/2, kernel lambda = (|u|^2 + |v|^2)/n.
Exact function-space updates (Eqs. 6-7):
    f_{t+1} = (1 - eta lambda_t + eta^2 f_t^2 / n) f_t,     lambda_{t+1} = lambda_t + eta f_t^2 / n (eta lambda_t - 4).
Phases: lazy eta < 2/lambda0, catapult 2/lambda0 < eta < 4/lambda0 (curvature ends below 2/eta), divergent eta > 4/lambda0.
"""
from common import *

print("lewkowycz2020large")
rng = np.random.default_rng(0)

# 1. Eqs. (6)-(7) are exact for the parameter-space model
for n in [1, 7, 200]:
    u, v = rng.normal(size=n), rng.normal(size=n)
    f = u @ v / np.sqrt(n); lam = (u @ u + v @ v) / n
    eta = 2.6 / lam; err = 0
    for t in range(40):
        fp = u @ v / np.sqrt(n)
        u, v = u - eta * fp * v / np.sqrt(n), v - eta * fp * u / np.sqrt(n)
        f, lam = (1 - eta * lam + eta ** 2 * f * f / n) * f, lam + eta * f * f / n * (eta * lam - 4)
        err = max(err, abs(f - u @ v / np.sqrt(n)) / (1 + abs(f)), abs(lam - (u @ u + v @ v) / n) / (1 + lam))
        if abs(f) > 1e6: break
    check(f"Eqs. (6)-(7) reproduce GD on u, v (n = {n})", err < 1e-9, f"max rel err {err:.1e}")


def fs(eta, lam0, f0, n, T=4000):
    f, lam, out = f0, lam0, []
    for t in range(T):
        out.append((t, f * f / 2, lam))
        f, lam = (1 - eta * lam + eta ** 2 * f * f / n) * f, lam + eta * f * f / n * (eta * lam - 4)
        if not np.isfinite(f) or abs(f) > 1e8:
            out.append((t + 1, np.inf, lam)); return np.array(out), "diverged"
    return np.array(out), "converged" if out[-1][1] < 1e-12 else "running"


# 2. phases at large width
n, lam0, f0 = 1000, 1.0, 1.0
for c, want in [(1.5, "lazy"), (2.5, "catapult"), (3.5, "catapult"), (4.3, "diverge")]:
    tr, st = fs(c / lam0, lam0, f0, n)
    peak = tr[:, 1].max(); lamf = tr[-1, 2]
    if want == "lazy":
        check(f"eta lambda0 = {c}: lazy (lambda changes < 1%, loss never grows)", st == "converged" and abs(lamf - lam0) < .01 and peak <= .5 + 1e-12, f"lambda_end = {lamf:.4f}")
    elif want == "catapult":
        check(f"eta lambda0 = {c}: catapult (loss spikes, lambda ends below 2/eta)", st == "converged" and peak > 10 and lamf < 2 / (c / lam0), f"peak loss {peak:.1f}, lambda_end = {lamf:.3f} < 2/eta = {2 * lam0 / c:.3f}")
    else:
        check(f"eta lambda0 = {c}: divergent", st == "diverged")
# catapult takes ~log(n) steps
steps = {}
for nn in [1e2, 1e4, 1e6]:
    tr, st = fs(3 / lam0, lam0, f0, nn)
    steps[nn] = int(np.argmax(tr[:, 1]))
check("time to the loss peak grows like log n", steps[1e4] - steps[1e2] > 3 and abs((steps[1e6] - steps[1e4]) - (steps[1e4] - steps[1e2])) <= 3, f"peak step {steps}")
# boundary exactly at 4/lambda0
tr1, s1 = fs(3.98, 1, 1, 1000, 20000); tr2, s2 = fs(4.02, 1, 1, 1000, 20000)
check("max learning rate = 4/lambda0 (3.98 converges, 4.02 diverges)", s1 == "converged" and s2 == "diverged")

# 3. 3D tab: n = 1 parameter model L(u, v) = (uv)^2/2, lambda = u^2 + v^2
L = lambda z: .5 * (z[0] * z[1]) ** 2
g = lambda z: np.array([z[0] * z[1] ** 2, z[1] * z[0] ** 2])
check_grad("n = 1 model (uv)^2/2", L, g, rand_pts(5, -2, 2))
check_hess_at("on the minimum (2, 0): eigenvalues 0 and u^2 = 4 = lambda", L, [2, 0], [[0, 0], [0, 4]])
u0 = np.array([2.0, .35]); lam0 = u0 @ u0
res = {}
for c in [1.6, 2.6, 3.4, 4.2]:
    z = u0.copy(); eta = c / lam0; Ls = []
    for t in range(3000):
        z = z - eta * g(z); Ls.append(L(z))
        if not np.all(np.isfinite(z)) or np.abs(z).max() > 1e3: break
    res[c] = (z @ z, max(Ls), Ls[-1], len(Ls))
    print(f"    n=1 eta*lambda0={c}: lambda_end={z @ z:.3f} 2/eta={2 / eta:.3f} peak loss={max(Ls):.3f} (start {L(u0):.3f}) end loss={Ls[-1]:.1e}")
check("n = 1, eta lambda0 = 1.6: converges without a loss spike", res[1.6][1] <= L(u0) + 1e-12 and res[1.6][2] < 1e-10)
check("n = 1, eta lambda0 = 2.6 and 3.4: loss spikes, ends flatter than 2/eta", all(res[c][1] > L(u0) and res[c][0] < 2 / (c / lam0) and res[c][2] < 1e-10 for c in [2.6, 3.4]))
check("n = 1, eta lambda0 = 4.2: diverges", res[4.2][3] < 3000)
# notes' teaching toy
Lw = lambda w: .5 * (1 - w[0] ** 2) ** 2
check_hess_at("notes toy L = (1 - w^2)^2/2: L'' = -2 + 6 w^2 at w = 0.7", Lw, [.7], [[-2 + 6 * .49]])
done()
