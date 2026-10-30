"""Mohtashami, Jaggi, Stich 2023: special properties of gradient descent with large learning rates.

Lemma 1 / Theorem 1: a sharp (mu_dagger-OPSC) local minimum x_dagger inside a landscape that is otherwise
one-point-strongly-convex towards the global minimum x*: with 2/mu_dagger < eta < mu*/L^2 GD cannot stay near x_dagger and
converges to x*; with a small eta there is positive probability of converging to x_dagger.
Theorem 2: the probability that randomly initialised GD never visits a small region X grows with the learning rate.
Toy: L(z) = B |z - c_g|^2 - A exp(-|z - c_s|^2 / (2 s^2)): a bowl around the global minimum with a narrow spurious well.
"""
from common import *

print("mohtashami2023special")
CG, CS, B, A, S = np.array([.8, -.4]), np.array([-.7, .5]), .15, .35, .15
L = lambda z: B * ((np.asarray(z) - CG) ** 2).sum() - A * np.exp(-((np.asarray(z) - CS) ** 2).sum() / (2 * S * S))
G = lambda z: 2 * B * (np.asarray(z) - CG) + A * np.exp(-((np.asarray(z) - CS) ** 2).sum() / (2 * S * S)) * (np.asarray(z) - CS) / S ** 2
check_grad("bowl + sharp well", L, G, rand_pts(6, -1.5, 1.5))
z = CS.copy()
for _ in range(20000): z = z - .01 * G(z)
zl = z; zg = CG.copy()
for _ in range(2000): zg = zg - .1 * G(zg)
lam_l, lam_g = eig(fd_hess(L, zl))[-1], eig(fd_hess(L, zg))[-1]
check("spurious local minimum is higher than the global one", L(zl) > L(zg) + .05, f"L(local) = {L(zl):.3f}, L(global) = {L(zg):.4f}")
check("local minimum is sharp, global minimum flat", lam_l > 10 * lam_g, f"lambda_max local {lam_l:.2f} (2/lambda = {2 / lam_l:.3f}), global {lam_g:.3f} (2/lambda = {2 / lam_g:.2f})")


def run(z0, eta, T=3000):
    z = np.array(z0, float)
    for _ in range(T):
        z = z - eta * G(z)
        if np.abs(z).max() > 1e3: return z, "div"
    return z, "global" if np.linalg.norm(z - zg) < 1e-3 else "local" if np.linalg.norm(z - zl) < 1e-3 else "other"


START = [-.55, .62]
for eta, want in [(.05, "local"), (.3, "global"), (.8, "global")]:
    zz, st = run(START, eta)
    check(f"from {START} with eta = {eta}: ends at the {want} minimum", st == want, f"{st}, end {np.round(zz, 3)}")
check("eta = 0.3 > 2/lambda_local: the sharp minimum is linearly unstable", .3 > 2 / lam_l)
# gradient norm increases temporarily during the escape (Sec. 1)
z = np.array(START, float); gn = []
for _ in range(60):
    gn.append(np.linalg.norm(G(z))); z = z - .3 * G(z)
check("escape with eta = 0.3: the gradient norm rises temporarily", max(gn[3:]) > gn[3] * 1.2, f"max |g| {max(gn):.2f} vs {gn[3]:.2f} at step 3")
# Theorem 2 flavour: fraction of random starts that end at the global minimum vs eta
rng = np.random.default_rng(0); P = rng.uniform(-1.6, 1.6, size=(600, 2))
etas = [.01, .02, .04, .06, .08, .1, .12, .15, .2, .3, .5, 1, 2, 4, 6]
frac = []
for eta in etas:
    res = [run(p, eta, 8000)[1] for p in P]
    frac.append([eta, round(np.mean([r == "global" for r in res]), 4), round(np.mean([r == "local" for r in res]), 4), round(np.mean([r == "div" for r in res]), 4)])
print("    eta, P(global), P(local), P(div):", frac)
fl = [f[2] for f in frac]
check("P(end at the spurious minimum) falls to 0 once eta > 2/lambda_local", all(f[2] == 0 for f in frac if f[0] > 2 / lam_l) and frac[0][2] > .03)
check("P(local) is non-increasing in eta (Theorem 2 trend)", all(fl[i] >= fl[i + 1] - .01 for i in range(len(fl) - 1)))
check("GD diverges only beyond 2/lambda_global (bowl curvature 2B = 0.3 -> 6.67)", all(f[3] == 0 for f in frac if f[0] < 2 / lam_g))
save_json("mohtashami2023special", "data.json", {"frac": frac, "lamL": round(float(lam_l), 3), "lamG": round(float(lam_g), 3), "zl": r(zl), "zg": r(zg)})
# notes example: a = 10, eta 0.1 -> multiplier 0; eta 0.25 -> -1.5
check("notes: 1 - 0.1*10 = 0 and 1 - 0.25*10 = -1.5", 1 - .1 * 10 == 0 and 1 - .25 * 10 == -1.5)
done()
