"""Allen-Zhu, Li, Song 2019: deep ReLU nets (fixed A, B; trained W_1..W_L) — width, gradient lower bound, separation, Fig. 1 slice.

Writes papers/allenzhu2019convergence/data.json.
"""
from common import *

print("allenzhu2019convergence")
Ld, d = 3, 10


def make(n, delta=None, seed=0):
    r = np.random.default_rng(seed)
    if delta is None:
        X = r.normal(size=(n, d))
        y = np.sign(r.normal(size=n))
    else:  # two unit inputs at distance delta, opposite labels
        e1, e2 = np.eye(d)[0], np.eye(d)[1]
        ang = 2 * np.arcsin(delta / 2)
        X = np.array([e1, np.cos(ang) * e1 + np.sin(ang) * e2])
        y = np.array([1.0, -1.0])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X, y


def init(m, seed=1):
    r = np.random.default_rng(seed)
    A = r.normal(0, np.sqrt(2 / m), (m, d))
    Ws = [r.normal(0, np.sqrt(2 / m), (m, m)) for _ in range(Ld)]
    B = r.normal(0, 1.0, m)
    return A, Ws, B


def fwd(A, Ws, B, X):
    hs = [np.maximum(X @ A.T, 0)]
    for W in Ws:
        hs.append(np.maximum(hs[-1] @ W.T, 0))
    return hs, hs[-1] @ B


def loss_grad(A, Ws, B, X, y):
    hs, out = fwd(A, Ws, B, X)
    e = out - y
    F = .5 * e @ e
    gs = [None] * Ld
    delta = np.outer(e, B) * (hs[-1] > 0)
    for l in range(Ld - 1, -1, -1):
        gs[l] = delta.T @ hs[l]
        if l > 0:
            delta = (delta @ Ws[l]) * (hs[l] > 0)
    return F, gs


def train(m, X, y, steps, eta_c=.3, seed=1, record=True):
    A, Ws, B = init(m, seed)
    W0 = [W.copy() for W in Ws]
    eta = eta_c / (m * len(X))
    out = []
    for k in range(steps):
        F, gs = loss_grad(A, Ws, B, X, y)
        gn2 = sum(np.sum(g * g) for g in gs)
        mv = np.sqrt(sum(np.sum((W - W_0) ** 2) for W, W_0 in zip(Ws, W0))) / np.sqrt(sum(np.sum(W_0 ** 2) for W_0 in W0))
        if record:
            out.append((F, gn2 / max(F, 1e-300) / m, mv))
        Ws = [W - eta * g for W, g in zip(Ws, gs)]
    return np.array(out), (A, Ws, B)


X, y = make(10)
A, Ws, B = init(32)
flat = lambda th: loss_grad(A, [th[i * 1024:(i + 1) * 1024].reshape(32, 32) for i in range(3)], B, X, y)[0]
th0 = np.concatenate([W.ravel() for W in Ws])
gan = np.concatenate([g.ravel() for g in loss_grad(A, Ws, B, X, y)[1]])
idx = np.random.default_rng(0).choice(th0.size, 30, replace=False)
fdv = []
for i in idx:
    e = np.zeros_like(th0); e[i] = 1e-6
    fdv.append((flat(th0 + e) - flat(th0 - e)) / 2e-6)
check("backprop gradient matches finite differences", np.allclose(gan[idx], fdv, rtol=1e-4, atol=1e-7))
curves = {}
for m in [16, 64, 256, 1024]:
    o, _ = train(m, X, y, 600)
    curves[str(m)] = {"loss": sig(o[:, 0]), "ratio": sig(o[:, 1]), "move": sig(o[:, 2])}
    print(f"  m={m:5d}: F0={o[0, 0]:.3f} F600={o[-1, 0]:.2e}  min ||grad||^2/(F m) {o[:, 1].min():.3f}  movement {o[-1, 2]:.4f}")
check("wider nets reach lower loss in the same iterations", curves["1024"]["loss"][-1] < curves["16"]["loss"][-1])
check("relative weight movement shrinks with width", curves["1024"]["move"][-1] < curves["64"]["move"][-1] < curves["16"]["move"][-1])
mins = [min(curves[k]["ratio"]) for k in ["64", "256", "1024"]]
check("Theorem 3 quantity ||grad||^2/(m F) stays bounded below and does not shrink with width", mins[0] > .1 and mins[0] <= mins[1] <= mins[2], str(np.round(mins, 3)))
sep = {}
for dl in [1.0, .3, .1, .03]:
    X2, y2 = make(2, delta=dl)
    o, _ = train(256, X2, y2, 3000, eta_c=.3)
    hit = int(np.argmax(o[:, 0] < 1e-3)) if np.any(o[:, 0] < 1e-3) else None
    sep[str(dl)] = {"loss": sig(o[::5, 0]), "hit": hit}
    print(f"  delta={dl}: iterations to F<1e-3: {hit}")
hits = [sep[k]["hit"] or 10 ** 9 for k in ["1.0", ".3" if ".3" in sep else "0.3", "0.1", "0.03"]]
check("smaller separation delta -> slower training", hits[0] <= hits[1] <= hits[2] <= hits[3] and hits[3] > hits[0])
# Fig. 1 analogue: slice along -grad and the most negative curvature direction at W_t (m=256, t=20)
m = 256
_, (A, Ws, B) = train(m, X, y, 20, record=False)
shape = [W.shape for W in Ws]
def unflat(t):
    out, i = [], 0
    for s in shape:
        out.append(t[i:i + s[0] * s[1]].reshape(s)); i += s[0] * s[1]
    return out
th = np.concatenate([W.ravel() for W in Ws])
Fg = lambda t: loss_grad(A, unflat(t), B, X, y)
g0 = np.concatenate([g.ravel() for g in Fg(th)[1]])
def hvp(v, h=1e-4):
    gp = np.concatenate([g.ravel() for g in Fg(th + h * v)[1]])
    gm = np.concatenate([g.ravel() for g in Fg(th - h * v)[1]])
    return (gp - gm) / (2 * h)
rr = np.random.default_rng(5)
v = rr.normal(size=th.size); v /= np.linalg.norm(v)
lam_top = 0
for _ in range(40):
    w = hvp(v); lam_top = v @ w; v = w / np.linalg.norm(w)
v = rr.normal(size=th.size); v /= np.linalg.norm(v)
for _ in range(150):  # power iteration on (lam_top I - H) -> most negative direction
    w = lam_top * v - hvp(v); v = w / np.linalg.norm(w)
lam_min = v @ hvp(v)
print(f"  at W_20: lambda_max ~ {lam_top:.3f}, lambda_min ~ {lam_min:.4f}")
check("negative curvature exists along the trajectory", lam_min < 0)
d1 = -g0 / np.linalg.norm(g0)
d2 = v - (v @ d1) * d1
d2 /= np.linalg.norm(d2)
F0 = Fg(th)[0]
s1 = 2 * F0 / np.linalg.norm(g0)          # step scale along the gradient
s2 = s1
Agrid = np.linspace(-.5, 1.5, 41) * s1
Bgrid = np.linspace(-1, 1, 41) * s2 * 3
grid = np.array([[Fg(th + a * d1 + b * d2)[0] for a in Agrid] for b in Bgrid])
print(f"  slice: F(W_t)={F0:.3f}, min along -grad {grid[20].min():.3f}, F along curvature dir at +-max {grid[0, 10]:.3f}/{grid[-1, 10]:.3f}")
check("moving along -grad decreases F a lot", grid[20].min() < .6 * F0)
save_json("allenzhu2019convergence", "data.json", {"widths": curves, "separation": sep,
          "slice": {"F": r(grid, 5), "a": [float(Agrid[0]), float(Agrid[-1])], "b": [float(Bgrid[0]), float(Bgrid[-1])], "n": 41,
                    "F0": float(F0), "lam_min": float(lam_min), "lam_max": float(lam_top)},
          "setup": {"L": Ld, "d": d, "n": 10}})
done()
