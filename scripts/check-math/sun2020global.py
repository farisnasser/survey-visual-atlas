"""Sun, Li, Liang, Ding, Srikant 2020 ("The global landscape of neural networks: an overview", IEEE SPM 37(5); arXiv 2007.01429).
Sec. IV-A toy: one sample, one neuron, F(v, w) = (y - v phi(w x))^2 with (x, y) = (1, 1).
Fig. 2: phi(z) = -(z - 1)^2 / 8: (v, w) = (1, 1) is a sub-optimal local minimum (not strict).
Prop. IV.1: F has no sub-optimal local minima iff every zero t of phi is not a local min or max of phi (tanh: none; -(z-1)^2/8: yes).
"""
from common import *

print("sun2020global")
acts = {"quad": (lambda z: -(z - 1) ** 2 / 8, lambda z: -(z - 1) / 4), "tanh": (np.tanh, lambda z: 1 / np.cosh(z) ** 2)}
for name, (phi, dphi) in acts.items():
    Fn = lambda p: (1 - p[0] * phi(p[1])) ** 2
    gn = lambda p: np.array([-2 * (1 - p[0] * phi(p[1])) * phi(p[1]), -2 * (1 - p[0] * phi(p[1])) * p[0] * dphi(p[1])])
    check_grad(f"F, phi = {name}", Fn, gn, rand_pts(20, -3, 3))
phi, dphi = acts["quad"]
Fq = lambda p: (1 - p[0] * phi(p[1])) ** 2
ring = min(Fq(np.array([1, 1]) + e * np.array([np.cos(t), np.sin(t)])) for e in np.linspace(1e-3, .3, 60) for t in np.linspace(0, 2 * np.pi, 721))
check("Fig. 2: (1, 1) is a local minimum with F = 1 (F >= 1 on a disc of radius 0.3), not global (min F = 0)", ring >= 1 - 1e-12 and Fq([1, 1]) == 1)
H = fd_hess(Fq, [1, 1]); ev = eig(H)
check("not strict: Hessian at (1, 1) has a zero eigenvalue (flat along v) and a positive one", abs(ev[0]) < 1e-6 and ev[1] > 0, f"{ev.round(4)}")
check("the whole half-line {(v, 1): v > 0} consists of such minima; for v < 0 the points are saddles (F < 1 nearby)", Fq([2.5, 1.1]) >= 1 and Fq([-1, 1.1]) < 1)
# GD from a grid: share ending at a spurious minimum (F ~ 1) vs global (F ~ 0)
def run(name, T=6000, eta=.05):
    phi, dphi = acts[name]; G = np.linspace(-3, 3, 41); V, W = np.meshgrid(G, G); P = np.stack([V.ravel(), W.ravel()], 1)
    for _ in range(T):
        e = 1 - P[:, 0] * phi(P[:, 1]); P = P - eta * np.stack([-2 * e * phi(P[:, 1]), -2 * e * P[:, 0] * dphi(P[:, 1])], 1)
    Fv = (1 - P[:, 0] * phi(P[:, 1])) ** 2
    return float(np.mean(Fv < 1e-6)), float(np.mean(np.abs(Fv - 1) < 1e-3)), float(np.mean((Fv >= 1e-6) & (np.abs(Fv - 1) >= 1e-3)))
res = {k: run(k) for k in acts}
for k, v in res.items(): print(f"  GD (step 0.05, 6000 steps) from a 41 x 41 grid on [-3, 3]^2, phi = {k}: global {v[0]:.3f}, stuck at F = 1 {v[1]:.3f}, other {v[2]:.3f}")
check("quadratic activation: a sizeable share of GD runs ends at the spurious level F = 1", res["quad"][1] > .1)
check("tanh: the only grid start that stays at F = 1 is (0, 0), a critical point that is a saddle, not a minimum (Prop. IV.1)", abs(res["tanh"][1] - 1 / 41 ** 2) < 1e-12 and (1 - .1 * np.tanh(.1)) ** 2 < 1)
save_json("sun2020global", "data.json", {"res": res})

# in-page notes (papers/sun2020global/notes.md): toy Hessian (Sec. 2), x^2 - y^4 (Sec. 3), primer numbers
fab = lambda p: .5 * (p[0] * p[1] - 1) ** 2
check_hess_at("notes: (ab-1)^2/2 at 0 has Hessian [[0,-1],[-1,0]]", fab, [0, 0], [[0, -1], [-1, 0]])
check("notes: its eigenvalues are -1, +1", np.allclose(np.linalg.eigvalsh([[0, -1], [-1, 0]]), [-1, 1]))
gxy = lambda p: p[0] ** 2 - p[1] ** 4
check_hess_at("notes: x^2 - y^4 at 0 has Hessian diag(2, 0)", gxy, [0, 0], [[2, 0], [0, 0]])
check("notes: g(0, y) < g(0, 0) for small y != 0", all(gxy([0, y]) < 0 for y in [1e-3, -.01, .1]))
cr = np.sort(np.roots([4, 0, -4, .5]).real); fq4 = lambda x: x ** 4 - 2 * x ** 2 + .5 * x
check("primer: x^4 - 2x^2 + 0.5x critical points ~ -1.057, 0.127, 0.930", np.allclose(cr, [-1.057, .127, .930], atol=5e-4))
check("primer: f(-1.057) ~ -1.515 (global), f(0.930) ~ -0.517 (local), both with f'' > 0", abs(fq4(cr[0]) + 1.515) < 5e-4 and abs(fq4(cr[2]) + .517) < 5e-4 and 12 * cr[0] ** 2 - 4 > 0 and 12 * cr[2] ** 2 - 4 > 0 and 12 * cr[1] ** 2 - 4 < 0)
xx = .5
for _ in range(5000):
    xx -= .01 * (4 * xx ** 3 - 4 * xx + .5)
check("primer: GD from x = 0.5 ends at the worse minimum 0.930", abs(xx - cr[2]) < 1e-6)
check_hess_at("primer: x^2 + y^4 at 0 also has Hessian diag(2, 0)", lambda p: p[0] ** 2 + p[1] ** 4, [0, 0], [[2, 0], [0, 0]])
pts = rand_pts(20000, -3, 3, seed=4)
check("primer: (ab-1)^2/2 >= 0.5 wherever ab <= 0", all(fab(p) >= .5 for p in pts if p[0] * p[1] <= 0))
xs4 = np.linspace(-2, 2, 4000001); sub = xs4[(xs4 ** 2 - 1) ** 2 <= .5]
check("primer: {(x^2-1)^2 <= 0.5} = [-1.31,-0.54] U [0.54,1.31]; barrier f(0) = 1",
      abs(sub.max() - 1.31) < 5e-3 and abs(sub[sub > 0].min() - .54) < 5e-3 and not np.any(np.abs(sub) < .5) and (0 ** 2 - 1) ** 2 == 1)
th4 = np.linspace(0, 2 * np.pi, 1000)
check("primer: (x^2 + y^2 - 1)^2 = 0 along the whole unit circle", np.allclose((np.cos(th4) ** 2 + np.sin(th4) ** 2 - 1) ** 2, 0))
done()
