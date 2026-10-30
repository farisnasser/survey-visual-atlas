"""Polyak 1964: some methods of speeding up the convergence of iteration methods (heavy-ball momentum).

Heavy ball: x_{t+1} = x_t - alpha grad f(x_t) + beta (x_t - x_{t-1}).
On a quadratic with Hessian eigenvalues in [mu, L], the optimal choice alpha = 4/(sqrt L + sqrt mu)^2,
beta = ((sqrt L - sqrt mu)/(sqrt L + sqrt mu))^2 gives contraction factor (sqrt kappa - 1)/(sqrt kappa + 1) per step,
versus (kappa - 1)/(kappa + 1) for GD with its best step 2/(L + mu).
(The 1964 paper is not openly available; formulas checked here are the standard ones the notes and survey Eq. (5) use.)
"""
from common import *

print("polyak1964some")


def rho_hb(lam, a, b):
    return max(abs(np.roots([1, -(1 + b - a * lam), b])))


for L, mu in [(5., .2), (100., 1.), (10., 1.)]:
    k = L / mu; a = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2; b = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    worst = max(rho_hb(l, a, b) for l in np.linspace(mu, L, 400))
    target = (np.sqrt(k) - 1) / (np.sqrt(k) + 1)
    check(f"kappa = {k:.0f}: worst spectral radius over [mu, L] = (sqrt k - 1)/(sqrt k + 1)", abs(worst - target) < 1e-6, f"{worst:.6f} vs {target:.6f}")
    gdr = max(abs(1 - 2 / (L + mu) * l) for l in (mu, L))
    check(f"kappa = {k:.0f}: GD with step 2/(L + mu) contracts by (k - 1)/(k + 1)", abs(gdr - (k - 1) / (k + 1)) < 1e-12)
# 3D tab: f = 1/2 (5 x^2 + 0.2 y^2), start (1.4, 1.8)
L, mu = 5., .2
A_ = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2; B_ = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
f = lambda z: .5 * (L * z[0] ** 2 + mu * z[1] ** 2); g = lambda z: np.array([L * z[0], mu * z[1]])
check_grad("quadratic", f, g, rand_pts(3, -2, 2))


def iters(kind, tol=1e-8, T=5000):
    z = np.array([1.4, 1.8]); zp = z.copy()
    for t in range(T):
        if f(z) < tol: return t
        if kind == "gd": z = z - 2 / (L + mu) * g(z)
        else: z, zp = z - A_ * g(z) + B_ * (z - zp), z
    return T


ng, nh = iters("gd"), iters("hb")
print(f"    kappa = 25, f < 1e-8: GD {ng} steps, heavy ball {nh} steps (alpha {A_:.4f}, beta {B_:.4f})")
check("heavy ball needs several times fewer steps than GD at kappa = 25", ng > 3 * nh)
# notes' saddle remark: with curvature -a the characteristic polynomial has p(1) = -alpha a < 0
a_, al, be = .3, .1, .9
p1 = 1 - (1 + be + al * a_) + be
check("notes: at negative curvature -a, p(1) = -alpha a < 0, so a root exceeds 1 (escape)", abs(p1 + al * a_) < 1e-12 and max(np.roots([1, -(1 + be + al * a_), be]).real) > 1)
save_json("polyak1964some", "data.json", {"alpha": round(float(A_), 5), "beta": round(float(B_), 5), "ngd": ng, "nhb": nh})
done()
