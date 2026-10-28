"""Kawaguchi 2016: strict saddle (1 hidden layer) vs bad saddle (2 hidden layers)."""
from common import *

print("kawaguchi2016deep")
strict = lambda p: (1 - p[0] * p[1]) ** 2
strict_g = lambda p: np.array([-2 * p[1] * (1 - p[0] * p[1]), -2 * p[0] * (1 - p[0] * p[1])])
bad = lambda p: (1 - p[0] * p[1] ** 2) ** 2
bad_g = lambda p: np.array([-2 * p[1] ** 2 * (1 - p[0] * p[1] ** 2), -4 * p[0] * p[1] * (1 - p[0] * p[1] ** 2)])
bumpy = lambda p: .22 * (p[0] ** 2 + p[1] ** 2) + .3 * (1 - np.cos(4 * p[0]) * np.cos(4 * p[1]))
bumpy_g = lambda p: np.array([.44 * p[0] + 1.2 * np.sin(4 * p[0]) * np.cos(4 * p[1]),
                              .44 * p[1] + 1.2 * np.cos(4 * p[0]) * np.sin(4 * p[1])])
P = rand_pts(40, -1.6, 1.6)
check_grad("strict", strict, strict_g, P)
check_grad("bad", bad, bad_g, P)
check_grad("bumpy", bumpy, bumpy_g, P)
H = check_hess_at("strict at 0", strict, [0, 0], [[0, -2], [-2, 0]])
check("strict saddle eigenvalues are +-2", np.allclose(eig(H), [-2, 2], atol=1e-3))
check_hess_at("bad at 0", bad, [0, 0], [[0, 0], [0, 0]])
t = 0.1
check("bad: still a saddle (third-order decrease along (t,t))", bad([t, t]) < 1 and bad([-t, t]) > 1)
chain = lambda w: (1 - w[0] * w[1] * w[2]) ** 2
check("bad = slice w3=w2 of the 3-weight chain", abs(chain([.3, .7, .7]) - bad([.3, .7])) < 1e-12)
ps = gd(strict_g, [.05, .04], .02, 5000)
pb = gd(bad_g, [.05, .04], .02, 5000)
ts = int(np.argmax([strict(p) < 1e-3 for p in ps]))
tb = int(np.argmax([bad(p) < 1e-3 for p in pb]))
print(f"  steps to loss<1e-3 from (0.05,0.04), lr 0.02: strict {ts}, bad {tb}")
check("bad saddle escapes ~3x slower from the reference start", tb >= 2.5 * ts and ts > 0, f"ratio {tb / ts:.2f}")
ps2 = gd(strict_g, [.01, .008], .02, 40000)
pb2 = gd(bad_g, [.01, .008], .02, 40000)
r2 = int(np.argmax([bad(p) < 1e-3 for p in pb2])) / int(np.argmax([strict(p) < 1e-3 for p in ps2]))
check("gap widens closer to the saddle", r2 > tb / ts, f"ratio from (0.01,0.008): {r2:.1f}")
ends = [bumpy(gd(bumpy_g, s, .02, 3000)[-1]) for s in rand_pts(30, -1.35, 1.35, seed=3)]
check("bumpy has non-global minima", max(ends) > 0.1 and min(ends) < 1e-6, f"final losses {min(ends):.3f}..{max(ends):.3f}")
done()
