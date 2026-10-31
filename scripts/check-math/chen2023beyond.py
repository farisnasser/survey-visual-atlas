"""Chen & Bruna 2023: beyond the edge of stability via two-step gradient updates.

Theorem 1 (1-D): at a local min x*, if f'''(x*) != 0 and 3 f'''^2 - f'' f'''' > 0, the two-step map F^2 has fixed points
near x* for some eta slightly above 2/f''(x*).
Theorem 2: f(x) = (x^2 - mu)^2 / 4, eta = K/mu with 1 < K < sqrt(4.5) - 1 ~ 1.121, x0 in (0, sqrt(mu)): GD converges to a period-2 orbit
(except a measure-zero set of starts).
Theorem 3: f(x, y) = (xy - mu)^2 / 2, eta = K/mu, 1 < K < 1.5, x, y > 0: |x - y| -> 0 along the iterates with xy > mu.
Prop. 3: for 1 < K < 1.121 the 2-D GD converges to the same period-2 orbit on the diagonal x = y.
On the minimum manifold xy = mu the top Hessian eigenvalue is (x - y)^2 + 2 mu, smallest (2 mu) at x = y = sqrt(mu).
"""
from common import *

print("chen2023beyond")
MU = 1.0
f1 = lambda x: (x * x - MU) ** 2 / 4
F = lambda x, eta: x - eta * x * (x * x - MU)
# notes' worked numbers (mu = 1, eta = 1.05, x0 = 0.8)
x1 = F(.8, 1.05); x2 = F(x1, 1.05)
check("notes: x1 = 1.1024, x2 ~ 0.8532", abs(x1 - 1.1024) < 1e-12 and abs(x2 - .8532) < 1e-4, f"x2 = {x2:.5f}")
# Theorem 1 condition for the quartic at x* = 1: f'' = 2, f''' = 6, f'''' = 6
d2, d3, d4 = 3 * 1 - 1, 6 * 1, 6
check("quartic: f''(1) = 2, f'''(1) = 6, f''''(1) = 6 (finite differences)", abs(fd_hess(lambda v: f1(v[0]), [1.])[0, 0] - 2) < 1e-6)
check("Theorem 1 condition 3 f'''^2 - f'' f'''' = 96 > 0", 3 * d3 ** 2 - d2 * d4 == 96)
# quadratic has no such orbit: f''' = f'''' = 0 -> condition fails (0 > 0 false); GD diverges beyond 2/a
check("quadratic fails the condition (3*0 - a*0 = 0)", not (3 * 0 - 2 * 0 > 0))


def late(K, x0=.5, T=4000, keep=64):
    x = x0
    for _ in range(T):
        x = F(x, K / MU)
        if abs(x) > 1e6: return None
    out = []
    for _ in range(keep):
        x = F(x, K / MU); out.append(x)
    return np.array(out)


def period(seq, tol=1e-9):
    for p in [1, 2, 4, 8, 16, 32]:
        if np.all(np.abs(seq[p:] - seq[:-p]) < tol): return p
    return 0


KC = np.sqrt(4.5) - 1
check("sqrt(4.5) - 1 = 1.1213", abs(KC - 1.12132) < 1e-4)
for K, want in [(.9, 1), (1.05, 2), (1.1, 2), (1.12, 2), (1.2, 2), (1.25, 4), (1.295, 8)]:
    s = late(K); p = period(s) if s is not None else -1
    check(f"1-D quartic, K = {K}: period {want}", p == want, f"found period {p}")
# the guaranteed range of Theorem 2 is sufficient, not sharp: period 2 persists until the first doubling at K ~ 1.237
Ks_ = np.arange(1.0, 1.45, .001); pers = [period(late(K)) if late(K) is not None else -1 for K in Ks_]
k4 = Ks_[pers.index(4)]
check("first period doubling (2 -> 4) happens beyond sqrt(4.5) - 1", k4 > KC, f"at K = {k4:.3f}")
K_PD = round(float(k4), 3)
# the period-2 orbit for K = 1.05 and its two-step stability
s = late(1.05); a, b = s[0], s[1]
dF = lambda x, eta: 1 - eta * (3 * x * x - MU)
mult = dF(a, 1.05) * dF(b, 1.05)
check("K = 1.05: orbit points straddle the minimum and |F'(a)F'(b)| < 1", min(a, b) < 1 < max(a, b) and abs(mult) < 1, f"orbit ({min(a, b):.4f}, {max(a, b):.4f}), multiplier {mult:.4f}")
check("each orbit point alone is beyond 1-step stability at the minimum: eta f''(1) = 2K > 2", 2 * 1.05 > 2)
# 2-D: f(x, y) = (xy - mu)^2 / 2
f2 = lambda z: (z[0] * z[1] - MU) ** 2 / 2
g2 = lambda z: np.array([(z[0] * z[1] - MU) * z[1], (z[0] * z[1] - MU) * z[0]])
check_grad("2-D (xy - mu)^2/2", f2, g2, rand_pts(5, .2, 2))
H = fd_hess(f2, [2., .5])
check("top Hessian eigenvalue on xy = mu equals (x - y)^2 + 2 mu", abs(eig(H)[-1] - ((2 - .5) ** 2 + 2)) < 1e-5, f"{eig(H)[-1]:.4f}")
Z0 = np.array([1.3, .5])          # Theorem 3 needs x, y > 0 throughout; strongly imbalanced starts (e.g. (1.6, 0.25)) send y negative


def run2(K, T=3000):
    z = Z0.copy(); out = []
    for t in range(T):
        z = z - K / MU * g2(z); out.append(z.copy())
        if np.abs(z).max() > 1e3: break
    return np.array(out)


for K in [1.05, 1.1]:
    tr = run2(K)
    check(f"2-D, K = {K}: x, y stay positive (Theorem 3's assumption)", tr.min() > 0)
    P = tr[tr[:, 0] * tr[:, 1] > MU]
    gap = np.abs(P[:, 0] - P[:, 1])
    lt = tr[-64:, 0]
    check(f"2-D, K = {K}: |x - y| -> 0 on the xy > mu iterates (Theorem 3)", gap[-1] < 1e-8 and gap[0] > .4, f"{gap[0]:.3f} -> {gap[-1]:.1e}")
    check(f"2-D, K = {K}: converges to the 1-D period-2 orbit on the diagonal (Prop. 3)", period(lt) == 2 and np.allclose(sorted(lt[:2]), sorted(late(K)[:2]), atol=1e-7), f"orbit {np.round(sorted(lt[:2]), 4)}")
tr = run2(.9)
check("2-D, K = 0.9 < 1: converges to a minimum with sharpness (x - y)^2 + 2 < 2/eta", abs(tr[-1, 0] * tr[-1, 1] - 1) < 1e-10 and (tr[-1, 0] - tr[-1, 1]) ** 2 + 2 < 2 / .9, f"end {np.round(tr[-1], 4)}, sharpness {(tr[-1, 0] - tr[-1, 1]) ** 2 + 2:.3f} < {2 / .9:.3f}")
# bifurcation data for the 2-D tab
Ks = np.linspace(.8, 1.45, 521); pts = []
for K in Ks:
    s = late(K, T=3000, keep=48)
    if s is None: continue
    pts += [[round(float(K), 4), round(float(v), 4)] for v in np.unique(np.round(s, 4))]
# in-page notes (papers/chen2023beyond/notes.md): primer numbers (mu = 1, eta = 1.05)
check("primer 1: multiplier at the minimum 1 - 1.05*2 = -1.1; 0.01 -> -0.011 -> 0.0121", abs(dF(1., 1.05) + 1.1) < 1e-12 and abs(.01 * 1.1 ** 2 - .0121) < 1e-12)
check("primer 2: f''(0.8) = 0.92, f''(1.1024) ~ 2.646", abs(3 * .8 ** 2 - 1 - .92) < 1e-12 and abs(3 * 1.1024 ** 2 - 1 - 2.646) < 1e-3)
x = .8
for _ in range(4000):
    x = F(x, 1.05)
oa, ob = sorted([x, F(x, 1.05)])
check("primer 3: from x0 = 0.8 the iterates settle on {0.8729, 1.0911}", abs(oa - .8729) < 1e-4 and abs(ob - 1.0911) < 1e-4, f"{oa:.5f}, {ob:.5f}")
check("primer 3: loss alternates ~0.0142 / ~0.0091", abs(f1(oa) - .0142) < 1e-4 and abs(f1(ob) - .0091) < 1e-4, f"{f1(oa):.5f}, {f1(ob):.5f}")
check("primer 3: two-step multiplier ~0.595", abs(dF(oa, 1.05) * dF(ob, 1.05) - .595) < 1e-3)
check("primer 4: 2-cycle lasts until eta ~ 1.237 (first period 4)", abs(K_PD - 1.237) < 2e-3, f"{K_PD}")
save_json("chen2023beyond", "data.json", {"bif": pts, "Kc": round(float(KC), 5), "Kpd": K_PD})
done()
