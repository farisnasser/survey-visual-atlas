"""Safran & Shamir 2018: closed-form population loss, GD experiment (paper protocol), slice through a spurious minimum.

Writes papers/safran2018spurious/data.json.
"""
from common import *

print("safran2018spurious")


def kern(W, V):
    """E[[w.x]_+ [v.x]_+] for x~N(0,I): |w||v|/(2pi) (sin t + (pi - t) cos t). W: (..., n, d), V: (..., m, d)."""
    nw = np.linalg.norm(W, axis=-1)[..., :, None]
    nv = np.linalg.norm(V, axis=-1)[..., None, :]
    c = np.clip((W @ np.swapaxes(V, -1, -2)) / np.maximum(nw * nv, 1e-300), -1, 1)
    t = np.arccos(c)
    return nw * nv / (2 * np.pi) * (np.sin(t) + (np.pi - t) * c)


def F(W, V):
    return .5 * (kern(W, W).sum((-1, -2)) - 2 * kern(W, V).sum((-1, -2)) + kern(V, V).sum((-1, -2)))


def gradF(W, V):
    """Analytic gradient of F w.r.t. W (paper Sec. 4.2): d/dw of g(w,v) = |v|/(2pi) [sin t w/|w| + (pi - t) v/|v|]."""
    def dk(W, U):
        nw = np.linalg.norm(W, axis=-1)[..., :, None]
        nu = np.linalg.norm(U, axis=-1)[..., None, :]
        c = np.clip((W @ np.swapaxes(U, -1, -2)) / np.maximum(nw * nu, 1e-300), -1, 1)
        t = np.arccos(c)
        a = nu * np.sin(t) / (2 * np.pi * np.maximum(nw, 1e-300))  # coefficient on w
        b = (np.pi - t) / (2 * np.pi)                             # coefficient on u
        return a.sum(-1)[..., None] * W + b @ U
    return dk(W, W) - dk(W, V)  # d/dw_i of 1/2 sum_ij g(w_i,w_j) = sum_j g'(w_i,w_j) ; symmetric terms


rng = np.random.default_rng(0)
# closed form vs Monte Carlo
k, n = 3, 3
V = np.eye(k)
W = rng.normal(0, .7, (n, k))
X = rng.normal(size=(400000, k))
mc = .5 * np.mean((np.maximum(X @ W.T, 0).sum(1) - np.maximum(X @ V.T, 0).sum(1)) ** 2)
check("closed-form F matches Monte Carlo", abs(F(W, V) - mc) < 3e-3, f"{F(W, V):.4f} vs {mc:.4f}")
fw = lambda w: F(w.reshape(n, k), V)
check_grad("F (analytic)", fw, lambda w: gradF(w.reshape(n, k), V).ravel(), [rng.normal(0, .7, n * k) for _ in range(10)], tol=1e-4)
check("global minimum F(V,V) = 0", abs(F(V, V)) < 1e-12)


def gd_batch(k, n, runs, lr=.1, max_steps=60000, tol=1e-7, seed=0):
    r = np.random.default_rng(seed)
    V = np.eye(k)
    W = r.normal(0, 1 / np.sqrt(k), (runs, n, k))  # Xavier as in the paper: N(0, I/k)
    for s in range(max_steps):
        G = gradF(W, V)
        W -= lr * G
        if s % 500 == 0 and np.max(np.linalg.norm(G, axis=-1)) < tol:
            break
    return W, F(W, V), np.max(np.linalg.norm(gradF(W, V), axis=-1), -1)


results = {}
for k, n, runs in [(6, 6, 600), (8, 8, 400), (10, 10, 300), (12, 12, 200), (10, 11, 300), (10, 12, 300)]:
    Wf, Fv, Gn = gd_batch(k, n, runs)
    conv = Gn < 1e-5
    spur = (Fv > 1e-3) & conv
    results[f"{k},{n}"] = {"k": k, "n": n, "runs": runs, "final": r(np.sort(Fv), 5), "spurious_frac": float(spur.mean()),
                           "unconverged": int((~conv).sum())}
    print(f"  k={k:2d} n={n:2d}: spurious {spur.mean() * 100:5.1f}%  (unconverged {(~conv).sum()}), mean spurious F {Fv[spur].mean() if spur.any() else 0:.4f}")
check("k=n: spurious fraction grows with width", results["6,6"]["spurious_frac"] < results["8,8"]["spurious_frac"] < results["12,12"]["spurious_frac"])
check("over-parameterization (n=k+2) reduces failures", results["10,12"]["spurious_frac"] < results["10,10"]["spurious_frac"])
# find one spurious minimum for k=n=8 and build a 2D slice toward the nearest global minimum
Wf, Fv, Gn = gd_batch(8, 8, 200, seed=3)
idx = [i for i in range(len(Fv)) if Fv[i] > 1e-3 and Gn[i] < 1e-5][0]
Ws = Wf[idx]
k = n = 8
V = np.eye(k)
H = fd_hess(lambda w: F(w.reshape(n, k), V), Ws.ravel(), h=1e-4)
lam_min = eig(H)[0]
print(f"  spurious min: F={F(Ws, V):.5f}, |grad|={np.linalg.norm(gradF(Ws, V)):.1e}, lambda_min(H)={lam_min:.5f}")
check("found point is a strict local minimum with positive loss", lam_min > 1e-4 and F(Ws, V) > 1e-3)
from itertools import permutations
# nearest global minimum: assign each student unit to a teacher vector (greedy on inner products)
C = Ws @ V.T
best_p = max(permutations(range(k)), key=lambda P: sum(C[i, P[i]] for i in range(k)))
best = V[list(best_p)]
d1 = (best - Ws).ravel()
d2 = rng.normal(size=d1.size)
d2 -= d2 @ d1 / (d1 @ d1) * d1
d2 *= np.linalg.norm(d1) / np.linalg.norm(d2)
A = np.linspace(-.35, 1.25, 81)
B = np.linspace(-.8, .8, 81)
grid = np.array([[F((Ws.ravel() + a * d1 + b * d2).reshape(n, k), V) for a in A] for b in B])
line = [F((Ws.ravel() + a * d1).reshape(n, k), V) for a in np.linspace(0, 1, 41)]
print(f"  slice: F at spurious {grid[40, 22]:.4f}, barrier on the straight path max {max(line):.4f}, at global {line[-1]:.1e}")
check("straight path from spurious to global min rises first (a barrier)", max(line) > F(Ws, V) + 1e-4)
save_json("safran2018spurious", "data.json", {
    "experiments": results,
    "paper_table1": {"6": .3, "7": 5.5, "8": 12.6, "9": 21.8, "10": 34.6, "11": 45.5, "12": 58.5, "13": 73, "14": 73.6, "15": 80.3, "16": 85.1, "17": 89.7, "18": 90, "19": 93.4, "20": 94},
    "slice": {"a": [float(A[0]), float(A[-1])], "b": [float(B[0]), float(B[-1])], "n": 81, "F": r(grid, 5),
              "spurious": [0, 0], "global": [1, 0], "F_spurious": float(F(Ws, V)), "lam_min": float(lam_min),
              "Ws": r(Ws, 8), "d1": r(d1.reshape(n, k), 8), "d2": r(d2.reshape(n, k), 8), "k": k},
})

# in-page notes (papers/safran2018spurious/notes.md): Sec. 3 example, table arithmetic and the primer's numbers
Rs = np.random.default_rng(55)
xg = Rs.normal(size=2_000_000)
check("notes Sec. 3: E[x^2 1{x>0}] = 1/2, F(0.6) = 0.04, F(1) = 0", abs(np.mean(xg ** 2 * (xg > 0)) - .5) < 2e-3 and np.isclose(.25 * (.6 - 1) ** 2, .04))
check("notes Sec. 6: percentages = runs/1000 (0.3, 34.6, 80.3, 94.0, 10.1, 49.1)",
      all(np.isclose(a / 10, b) for a, b in [(3, .3), (346, 34.6), (803, 80.3), (940, 94.0), (101, 10.1), (491, 49.1)]))
def Kp(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    th = np.arccos(np.clip(a @ b / (na * nb), -1, 1))
    return na * nb / (2 * np.pi) * (np.sin(th) + (np.pi - th) * np.cos(th))
e1, e2 = np.eye(2)
X2 = Rs.normal(size=(1_000_000, 2))
mc = lambda a, b: np.mean(np.maximum(X2 @ a, 0) * np.maximum(X2 @ b, 0))
check("primer idea 1: kernel = 1/2 at theta = 0, 1/(2 pi) ~ 0.159 at theta = pi/2 (and matches Monte Carlo)",
      np.isclose(Kp(e1, e1), .5) and round(Kp(e1, e2), 3) == .159 and abs(mc(e1, e1) - .5) < 3e-3 and abs(mc(e1, e2) - Kp(e1, e2)) < 3e-3)
F2 = lambda W, V: .5 * (sum(Kp(a, b) for a in W for b in W) - 2 * sum(Kp(a, b) for a in W for b in V) + sum(Kp(a, b) for a in V for b in V))
mid = (e1 + e2) / 2
import math
check("primer idea 2: permuted students zero loss, midpoint loss ~ 0.091",
      math.factorial(6) == 720 and abs(F2([e1, e2], [e1, e2])) < 1e-12 and abs(F2([e2, e1], [e1, e2])) < 1e-12 and round(F2([mid, mid], [e1, e2]), 3) == .091)
gb = lambda t: -.001 * t + .05 * t * t - t ** 3 / 6
check("primer idea 3: Taylor lower bound ~ +5.4e-5 at t = 0.05, negative at t = 0.01", round(gb(.05) * 1e5, 1) == 5.4 and gb(.01) < 0)
se = lambda h: 100 * np.sqrt(h / 1000 * (1 - h / 1000) / 1000)
check("primer idea 4: standard errors 0.17, 1.5, 0.75 percentage points", round(se(3), 2) == .17 and round(se(346), 1) == 1.5 and round(se(940), 2) == .75)
check("primer idea 5: [w.x/2]_+ * 2 = [w.x]_+", np.allclose(2 * np.maximum(X2[:1000] @ (np.array([.3, -1.2]) / 2), 0), np.maximum(X2[:1000] @ np.array([.3, -1.2]), 0)))
done()
