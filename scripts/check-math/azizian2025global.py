"""Azizian et al. 2025: global convergence time of SGD on the paper's modified three-hump camel (Fig. 1).

Checks the function, its five critical points, the Gaussian-action transition costs 2*df/sigma^2, the
forest/pruning energy E(Q|x), the notes' 3-state example, and replicates the scaling log E[tau] ~ E/eta offline.
"""
from common import *

print("azizian2025global")
f = lambda p: 2 * p[0] ** 6 / 13 + p[0] ** 5 / 8 - 91 * p[0] ** 4 / 64 - p[0] ** 3 / 2 + 42 * p[0] ** 2 / 16 + 5 * p[1] ** 2 / 4 + p[0] * p[1]
g = lambda p: np.array([12 * p[0] ** 5 / 13 + 5 * p[0] ** 4 / 8 - 91 * p[0] ** 3 / 16 - 1.5 * p[0] ** 2 + 42 * p[0] / 8 + p[1], 2.5 * p[1] + p[0]])
hs = lambda p: np.array([[60 * p[0] ** 4 / 13 + 2.5 * p[0] ** 3 - 273 * p[0] ** 2 / 16 - 3 * p[0] + 42 / 8, 1], [1, 2.5]])
check_grad("camel", f, g, rand_pts(40, -3, 3))
# critical points: y = -2x/5 and a quintic in x
roots = np.roots([12 / 13, 5 / 8, -91 / 16, -1.5, 42 / 8 - .4, 0])
xs = sorted(roots[np.abs(roots.imag) < 1e-9].real)
crit = [np.array([x, -.4 * x]) for x in xs]
check("five critical points", len(crit) == 5)
kinds = []
for c in crit:
    ev = eig(fd_hess(f, c)); check_hess_at(f"at ({c[0]:.3f},{c[1]:.3f})", f, c, hs(c), tol=2e-3)
    kinds.append("min" if ev[0] > 0 else "saddle")
    print(f"  ({c[0]:.4f}, {c[1]:.4f}) f = {f(c):.4f} eig {np.round(ev, 3)} {kinds[-1]}")
check("pattern min, saddle, min, saddle, min", kinds == ["min", "saddle", "min", "saddle", "min"])
p1, p2, p3, p4, p5 = crit
check("global minimum is the leftmost point (paper's p1)", f(p1) < f(p5) < f(p3))
print(f"  paper prints p1 ~ (-2.573, 1.029); the printed formula gives ({p1[0]:.3f}, {p1[1]:.3f})")

# Gaussian action: climbing df costs 2 df / sigma^2 (per unit of 1/eta). Edge costs between minima (units of 1/sigma^2):
B = {("p3", "p1"): 2 * (f(p2) - f(p3)), ("p3", "p5"): 2 * (f(p4) - f(p3)), ("p5", "p3"): 2 * (f(p4) - f(p5))}
B[("p5", "p1")] = B[("p5", "p3")] + B[("p3", "p1")]
for k, v in B.items():
    print(f"  B{k} = {v:.4f} / sigma^2")


def energies(B, nodes, target):
    """Brute-force forests (one out-edge per non-target node, all reach target, no cycles) and prunings."""
    import itertools
    edges = [e for e in B if e[0] in nodes]
    outs = {n: [e for e in edges if e[0] == n] for n in nodes}

    def reaches(sel, start):
        seen, cur = set(), start
        while cur != target:
            if cur in seen:
                return False
            seen.add(cur)
            nxt = [e[1] for e in sel if e[0] == cur]
            if not nxt:
                return False
            cur = nxt[0]
        return True
    forests = [sel for sel in itertools.product(*[outs[n] for n in nodes]) if all(reaches(sel, n) for n in nodes)]
    fmin = min(sum(B[e] for e in s) for s in forests)
    res = {}
    for p in nodes:
        pr = [e for e in edges if not reaches([e], p)]
        res[p] = fmin - min(B[e] for e in pr)
    return fmin, res


fmin, E = energies(B, ["p3", "p5"], "p1")
Ex = max(E[p] - (0 if p == "p3" else B[("p3", p)]) for p in ["p3", "p5"])
print(f"  min forest {fmin:.4f}, E(Q|p3) = {E['p3']:.4f}, E(Q|p5) = {E['p5']:.4f}, E(Q|x=p3) = {Ex:.4f}  (/sigma^2)")
check("E(Q|p3) = B31 + B53 - B35 (rare detour into p5 dominates)", abs(Ex - (B[("p3", "p1")] + B[("p5", "p3")] - B[("p3", "p5")])) < 1e-12)

# notes' 3-state example
B3 = {("A", "G"): 2, ("A", "B"): 3, ("B", "A"): 5, ("B", "G"): 8}
fm3, E3 = energies(B3, ["A", "B"], "G")
check("notes: min forest 7, E(G|A) = 4", fm3 == 7 and E3["A"] == 4)
for eta, want in [(.5, 3027.17), (.25, 8889035.88)]:
    u, v, w, z = [np.exp(-c / eta) for c in (2, 3, 5, 8)]
    T = (w + z + v) / (u * w + u * z + v * z)
    check(f"notes: exact T at eta={eta}", abs(T - want) / want < 1e-6, f"T = {T:.2f}, eta log T = {eta * np.log(T):.5f}")

# offline replication: hitting time of the delta-ball around p1 from p3, Gaussian noise sigma, steps eta
sigma, delta, ntraj = 11.0, .1, 400
Eabs = Ex / sigma ** 2
rng = np.random.default_rng(7)
rows = []
for eta in [.02, .017, .0145, .0125, .011, .0098]:
    x = np.tile(p3, (ntraj, 1)).astype(float)
    tau = np.full(ntraj, -1)
    alive = np.ones(ntraj, bool)
    n = 0
    while alive.any() and n < 4_000_000:
        xa = x[alive]
        gx = 12 * xa[:, 0] ** 5 / 13 + 5 * xa[:, 0] ** 4 / 8 - 91 * xa[:, 0] ** 3 / 16 - 1.5 * xa[:, 0] ** 2 + 42 * xa[:, 0] / 8 + xa[:, 1]
        gy = 2.5 * xa[:, 1] + xa[:, 0]
        xa = xa - eta * (np.column_stack([gx, gy]) + sigma * rng.normal(size=xa.shape))
        x[alive] = xa
        n += 1
        hit = np.zeros(ntraj, bool)
        hit[alive] = np.hypot(xa[:, 0] - p1[0], xa[:, 1] - p1[1]) < delta
        tau[hit] = n
        alive &= ~hit
    check(f"eta={eta}: no blow-up, all hit", np.isfinite(x).all() and (tau > 0).all(), f"{(tau > 0).mean():.2f} hit")
    m = tau[tau > 0].mean()
    rows.append([eta, float(np.log(m)), float(np.median(tau[tau > 0]))])
    print(f"  eta={eta}: mean tau {m:.0f}, log {np.log(m):.3f}, E/eta = {Eabs / eta:.3f}")
inv = np.array([1 / r_[0] for r_ in rows]); lg = np.array([r_[1] for r_ in rows])
slope_n = np.polyfit(inv, lg, 1)[0]
# SGD "time" eta*tau removes the 1/eta prefactor of an iteration count (a subexponential factor inside the theorem's +-a)
slope, icpt = np.polyfit(inv, lg + np.log(inv ** -1), 1)
print(f"  fitted slope of log E[tau]: {slope_n:.4f}; of log(eta E[tau]): {slope:.4f}; predicted E = {Eabs:.4f}")
check("log E[tau] slope within 25% of E (iteration count)", abs(slope_n - Eabs) / Eabs < .25)
check("log(eta E[tau]) slope within 5% of E", abs(slope - Eabs) / Eabs < .05)
save_json("azizian2025global", "data.json", {
    "crit": [[float(c[0]), float(c[1]), float(f(c))] for c in crit], "kinds": kinds,
    "B": {f"{a}-{b}": float(v) for (a, b), v in B.items()}, "E": float(Ex),
    "sigma": sigma, "delta": delta, "ntraj": ntraj, "rows": [[r_[0], round(r_[1], 4), r_[2]] for r_ in rows], "slope": round(float(slope), 4), "slope_iter": round(float(slope_n), 4), "icpt": round(float(icpt), 4)})

# in-page notes (papers/azizian2025global/notes.md): notes Sec. 3 graph enumeration, closure remark, primer numbers
import itertools
from math import comb
def forest_prune(C):
    out = {"A": [e for e in C if e[0] == "A"], "B": [e for e in C if e[0] == "B"]}
    reach = lambda S, s0: (lambda nx: nx == "G" or (nx is not None and nx != s0 and reach_step(S, nx, {s0})))(S.get(s0))
    def reach_step(S, n, seen):
        if n == "G": return True
        if n in seen or n not in S: return False
        return reach_step(S, S[n], seen | {n})
    forests = [C[(a, x)] + C[(b, y)] for (a, x), (b, y) in itertools.product(out["A"], out["B"]) if reach_step({"A": x, "B": y}, "A", set()) and reach_step({"A": x, "B": y}, "B", set())]
    prunes = [C[e] for e in C if not reach_step({e[0]: e[1]}, "A", set())]
    return sorted(forests), sorted(prunes)
C3 = {("A", "G"): 2, ("A", "B"): 3, ("B", "A"): 5, ("B", "G"): 8}
fo, pr = forest_prune(C3)
check("notes Sec. 3: forest costs 7, 10, 11; pruning costs 3, 5, 8; E(G|A) = 7 - 3 = 4", fo == [7, 10, 11] and pr == [3, 5, 8] and min(fo) - min(pr) == 4)
C3c = dict(C3); C3c[("B", "G")] = min(8, 5 + 2); C3c[("A", "G")] = min(2, 3 + 8)     # minimum-path closure
fo2, pr2 = forest_prune(C3c)
check("notes Sec. 3: minimum-path closure leaves the exponent 4 unchanged", min(fo2) - min(pr2) == 4, f"closed costs {C3c}")
for et in [.5, .25]:
    u, v, w, z = [np.exp(-c / et) for c in (2, 3, 5, 8)]
    TA = (w + z + v) / (u * w + u * z + v * z)
    TA_ls, TB_ls = np.linalg.solve([[u + v, -v], [-w, w + z]], [1, 1])
    check(f"notes Sec. 3: T_A closed form = linear solve at eta = {et}", abs(TA / TA_ls - 1) < 1e-9 and abs(v / (u + v) / np.exp(-1 / et) - 1) < .2 and abs(1 / (w + z) / np.exp(5 / et) - 1) < .2)
check("primer: e^4 ~ 54.6, e^8 ~ 2981, mixture mean 0.99*10 + 0.01*1e6 ~ 10,010", abs(np.exp(4) - 54.6) < .05 and abs(np.exp(8) - 2981) < 1 and abs(.99 * 10 + .01 * 1e6 - 10010) < 1)
Irate = .75 * np.log(1.5) + .25 * np.log(.5)
rates = [-np.log(sum(comb(n, k) for k in range((3 * n + 3) // 4, n + 1)) / 2 ** n) / n for n in (100, 400, 1600)]
check("primer: coin rate I ~ 0.131; -log P/n ~ 0.151, 0.138, 0.133 for n = 100, 400, 1600", abs(Irate - .131) < 5e-4 and np.allclose(rates, [.151, .138, .133], atol=6e-4), f"{np.round(rates, 4)}")
sg2 = 1.7; gvec = np.array([.4, -1.1]); vv = np.array([.3, .8])
Lg = lambda v_: np.sum((v_ + gvec) ** 2) / (2 * sg2)
check("notes Sec. 7: L(v) - L(-v) = 2 v.grad f / sigma^2; L(-grad f) = 0, L(grad f) = 2|grad f|^2/sigma^2", np.isclose(Lg(vv) - Lg(-vv), 2 * vv @ gvec / sg2) and np.isclose(Lg(-gvec), 0) and np.isclose(Lg(gvec), 2 * gvec @ gvec / sg2))
done()
