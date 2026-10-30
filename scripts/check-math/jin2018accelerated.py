"""Jin, Netrapalli, Jordan 2018: perturbed accelerated GD (Alg. 2) with negative curvature exploitation (Alg. 3).

Toy: f(x, y) = x^2/2 - (lam/2) y^2 + (lam/4) y^4  (strict saddle with weak curvature -lam, minima y = +-1, f = -lam/4).
Parameters as Eq. (3) with ell = 1 and the target curvature sqrt(rho eps) = lam: eta = 1/4, theta = sqrt(lam)/4,
gamma = theta^2/eta, s = gamma/(4 rho) with rho = 6 lam (|f'''| bound on the box |y| <= 1).
"""
from common import *

print("jin2018accelerated")


def mk(lam):
    f = lambda p: .5 * p[0] ** 2 - .5 * lam * p[1] ** 2 + .25 * lam * p[1] ** 4
    g = lambda p: np.array([p[0], -lam * p[1] + lam * p[1] ** 3])
    return f, g


f, g = mk(.1)
check_grad("toy", f, g, rand_pts(30, -1.5, 1.5))
check_hess_at("saddle", f, [0, 0], [[1, 0], [0, -.1]])
check_hess_at("min", f, [0, 1], [[1, 0], [0, .2]])


def escape(lam, method, y0=1e-4, T=200000):
    f, g = mk(lam)
    eta, th = .25, np.sqrt(lam) / 4
    x = np.array([1.0, y0]); v = np.zeros(2)
    for t in range(T):
        if abs(x[1]) > .5:
            return t
        if method == "gd":
            x = x - eta * g(x)
        else:
            y = x + (1 - th) * v
            xn = y - eta * g(y); v = xn - x; x = xn
    return T


lams = [.2, .1, .05, .02, .01]
sg = [escape(l, "gd") for l in lams]; sa = [escape(l, "agd") for l in lams]
for l, a, b in zip(lams, sg, sa):
    print(f"  lam={l}: GD escapes in {a} steps, AGD in {b}")
pg = np.polyfit(np.log(lams), np.log(sg), 1)[0]; pa = np.polyfit(np.log(lams), np.log(sa), 1)[0]
check("GD escape time ~ 1/lam (slope ~ -1)", abs(pg + 1) < .15, f"slope {pg:.3f}")
check("AGD escape time ~ 1/sqrt(lam) (slope ~ -1/2)", abs(pa + .5) < .15, f"slope {pa:.3f}")
check("AGD faster for every lam", all(b < a for a, b in zip(sg, sa)))


def pagd(lam, T=4000, seed=0, eps=1e-3, r=1e-3, Tp=60):
    """Algorithm 2 exactly (perturbation, AGD, NCE). Returns history of (f, E, nce flag)."""
    f, g = mk(lam)
    R = np.random.default_rng(seed)
    eta, th = .25, np.sqrt(lam) / 4
    gam = th ** 2 / eta; rho = 6 * lam; s = gam / (4 * rho)
    x = np.array([1.0, 0.0]); v = np.zeros(2); last = -10 ** 9; H = []
    for t in range(T):
        if np.linalg.norm(g(x)) <= eps and t - last > Tp:
            a = R.uniform(0, 2 * np.pi); x = x + r * np.sqrt(R.uniform()) * np.array([np.cos(a), np.sin(a)]); last = t
        E0 = f(x) + (v @ v) / (2 * eta)   # Hamiltonian after any kick, before the step
        y = x + (1 - th) * v
        xn = y - eta * g(y); vn = xn - x
        # (2) holds trivially (with equality) when v = 0, i.e. x = y; Alg. 3 then divides by |v| = 0, so we test (2) only for v != 0
        nce = np.any(v != 0) and f(x) <= f(y) + g(y) @ (x - y) - gam / 2 * np.sum((x - y) ** 2)
        if nce:
            if np.linalg.norm(v) >= s:
                xn = x.copy()
            else:
                dl = s * v / max(np.linalg.norm(v), 1e-300)
                xn = min([x + dl, x - dl], key=f)
            vn = np.zeros(2)
        x, v = xn, vn
        H.append((f(x), f(x) + (v @ v) / (2 * eta), nce, E0))
    return x, H


x, H = pagd(.1)
fH = np.array([h[0] for h in H]); EH = np.array([h[1] for h in H]); nc = np.array([h[2] for h in H])
check("PAGD from the saddle's stable set reaches a minimum", abs(f(x) + .025) < 1e-6, f"f = {f(x):.6f}")
E0 = np.array([h[3] for h in H])
ups_f = int((np.diff(fH) > 1e-15).sum()); ups_E = int(((EH - E0) > 1e-13).sum())
nce_ok = bool(np.all((EH - E0)[nc] < 0))
print(f"  f increased on {ups_f} steps; NCE fired {int(nc.sum())} times; E_(t+1) > E_t on {ups_E} steps")
check("f alone is not monotone along AGD", ups_f > 0)
check("Hamiltonian E = f + |v|^2/(2 eta) never increases over a step (Lemma 4 without NCE, Lemma 5 with NCE); only kicks add energy", ups_E == 0 and nce_ok)
save_json("jin2018accelerated", "data.json", {"lams": lams, "gd": sg, "agd": sa, "slopes": [round(pg, 3), round(pa, 3)]})
done()
