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

# in-page notes (papers/polyak1964some/notes.md): Sec. 3 tuning and rate (Polyak 1964, Theorem 9(3), p. 800 of the Russian
# original: q2 = (sqrt M - sqrt m)/(sqrt M + sqrt m) at alpha = 4/(sqrt M + sqrt m)^2, beta = q2^2), and the primer's numbers
for kap in (4., 25., 100.):
    Lk, mk = kap, 1.
    et = 4 / (np.sqrt(Lk) + np.sqrt(mk)) ** 2; bt = ((np.sqrt(Lk) - np.sqrt(mk)) / (np.sqrt(Lk) + np.sqrt(mk))) ** 2
    rad = max(max(abs(np.roots([1, -(1 + bt - et * lam), bt]))) for lam in np.linspace(mk, Lk, 2001))
    check(f"notes Sec. 3: kappa={kap:.0f}: worst root modulus = (sqrt k - 1)/(sqrt k + 1) = sqrt(beta)",
          abs(rad - (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)) < 1e-6 and abs(rad - np.sqrt(bt)) < 1e-6)
    gd_rad = max(abs(1 - 2 / (Lk + mk) * lam) for lam in np.linspace(mk, Lk, 2001))
    check(f"primer idea 1: kappa={kap:.0f}: GD with 2/(L+mu) has worst multiplier (k-1)/(k+1)", abs(gd_rad - (kap - 1) / (kap + 1)) < 1e-9)
q_gd, q_hb = 99 / 101, 9 / 11
n_gd, n_hb = np.log(1e-6) / np.log(q_gd), np.log(1e-6) / np.log(q_hb)
print(f"  kappa = 100: GD multiplier {q_gd:.4f} -> {n_gd:.1f} steps; heavy ball {q_hb:.4f} -> {n_hb:.1f} steps for a factor 1e6")
check("primer: kappa=100 -> 0.980 / ~690 steps (GD) vs 0.818 / ~69 steps (heavy ball)",
      round(q_gd, 3) == .980 and round(q_hb, 3) == .818 and abs(n_gd - 690) < 2 and abs(n_hb - 69) < 1)
check("primer idea 2: r^2 - 0.9 r + 0.2 has roots 0.5, 0.4", np.allclose(sorted(np.roots([1, -.9, .2])), [.4, .5]))
# idea 3: complex roots have modulus sqrt(b) whatever a is (a^2 < 4b)
check("primer idea 3: complex roots have |r| = sqrt(b)", all(np.allclose(abs(np.roots([1, -a3, .5])), np.sqrt(.5)) for a3 in np.linspace(-1.4, 1.4, 15)))
rts = np.roots([1, -(1 + .9 + .1 * 1), .9])
check("primer idea 5: eta=0.1, beta=0.9, curvature -1: roots 1 +- sqrt(0.1), growth ~1.316 vs GD 1.1",
      np.allclose(sorted(rts), [1 - np.sqrt(.1), 1 + np.sqrt(.1)]) and round(max(rts), 3) == 1.316 and np.isclose(1 + .1 * 1, 1.1))
done()
