"""Wu, Simsek, Ged 2025: one-neuron kink example + replication of the Fig. 3 experiment (Sec. 4.1.1).

Writes papers/wu2025loss/data.json.
"""
from common import *

print("wu2025loss")
# one neuron: y_hat = a * relu(w), x=(1,0), y=1  ->  L(w, a) = 1/2 (a relu(w) - 1)^2
L1 = lambda p: .5 * (p[1] * max(p[0], 0) - 1) ** 2
def L1g(p):
    w, a = p
    on = 1.0 if w > 0 else 0.0
    rres = a * max(w, 0) - 1
    return np.array([rres * a * on, rres * max(w, 0)])
check_grad("one neuron (w>0 half)", L1, L1g, rand_pts(20, .1, 1.6))
check_grad("one neuron (w<0 half)", L1, L1g, np.column_stack([rand_pts(20, -1.6, -.1)[:, 0], rand_pts(20, -1.6, 1.6, seed=4)[:, 1]]))
t = .05
check("origin: no first-order decrease, second-order decrease along (t,t)", L1([t, 0]) == .5 and L1([0, t]) == .5 and L1([t, t]) < .5)
check("dead half-plane w<0 is flat at L=1/2 (zero gradient)", L1g([-.5, .8]).tolist() == [0, 0])
p = gd(L1g, [.06, .05], .05, 2000)
check("small living neuron escapes to a*w = 1", abs(p[-1][0] * p[-1][1] - 1) < 1e-3)
c = p[:, 0] ** 2 - p[:, 1] ** 2
check("||w||^2 - a^2 nearly conserved (Fact 4.5, small steps)", np.max(np.abs(c - c[0])) < 5e-3)
# --- Fig. 3 replication ---
xs = np.array([-1, -.6, -.1, .3, .7])
ys = np.array([.28, -.1, .03, .23, -.22])
X = np.column_stack([xs, np.ones(5)])       # x_k = (x_k, 1)
H = 50
rng = np.random.default_rng(0)
W = rng.normal(0, 5e-6, (H, 2))
h = rng.normal(0, 5e-6, H)
lr, epochs = 1e-3, 500000
snap_at = np.unique(np.concatenate([np.linspace(0, epochs, 501).astype(int)]))
loss_curve, snaps = [], []
cons0 = np.sum(W ** 2, 1) - h ** 2
for e in range(epochs + 1):
    Z = X @ W.T                          # (5, H)
    A = np.maximum(Z, 0)
    pred = A @ h
    res = pred - ys
    if e % 250 == 0:
        loss_curve.append([e, float(.5 * np.sum(res ** 2))])
    if e in snap_at:
        snaps.append([e, r(np.arctan2(W[:, 1], W[:, 0]) % (2 * np.pi), 4), r(np.linalg.norm(W, axis=1), 6), r(np.abs(h), 6)])
    gh = A.T @ res
    gW = ((res[:, None] * (Z > 0)) * h[None, :]).T @ X
    h -= lr * gh
    W -= lr * gW
lc = np.array(loss_curve)
cons = np.sum(W ** 2, 1) - h ** 2
print(f"  final loss {lc[-1, 1]:.5f}; conservation drift max {np.max(np.abs(cons - cons0)):.2e}")
d = np.abs(np.diff(np.log(lc[:, 1])))
drops = lc[1:, 0][d > np.percentile(d, 99.3)]
print(f"  fastest loss drops near epochs: {np.unique((drops // 1000) * 1000)[:12]}")
rel = np.max(np.abs(cons - cons0)) / np.max(np.sum(W ** 2, 1))
check("||w_i||^2 - h_i^2 conserved up to step-size error (Fact 4.5)", rel < 1e-3, f"relative drift {rel:.1e}")
at = lambda e: lc[np.argmin(np.abs(lc[:, 0] - e)), 1]
plateaus = [at(10000), at(45000), at(300000)]
print(f"  plateau levels {np.round(plateaus, 5)}")
check("three plateaus: origin saddle, intermediate saddle, final local min", plateaus[0] - plateaus[1] > 5e-3 and plateaus[1] - plateaus[2] > 2e-3
      and abs(at(10000) - at(0)) < 1e-4 and abs(at(40000) - at(50000)) < 1e-4 and abs(at(300000) - at(500000)) < 1e-7)
ang = np.arctan2(W[:, 1], W[:, 0]) % (2 * np.pi)
dead = [i for i in range(H) if np.all(X @ W[i] <= 0)]
print(f"  dead neurons {len(dead)}; their angles in [{ang[dead].min() if dead else 0:.2f}, {ang[dead].max() if dead else 0:.2f}] (paper: 3.93-5.67)")
check("dead neurons sit in the paper's dead wedge (3.93, 5.67) rad", all(3.92 < ang[i] < 5.68 for i in dead))
big = np.argsort(-np.abs(h))[:6]
print(f"  largest neurons: angles {np.round(ang[big], 2)}, |h| {np.round(np.abs(h[big]), 3)}")
save_json("wu2025loss", "data.json", {"x": xs.tolist(), "y": ys.tolist(), "loss": r(lc, 7), "snaps": snaps,
                                       "dead_wedge": [3.93, 5.67], "lr": lr, "init_std": 5e-6, "H": H})

# in-page notes (papers/wu2025loss/notes.md): Sec. 2 gradient formula, Sec. 4 splitting identity and the primer's numbers
Rw = np.random.default_rng(56)
Xw, yw = Rw.normal(size=(7, 3)), Rw.normal(size=7)
aw, Ww = Rw.normal(size=4), Rw.normal(size=(4, 3))
Lw = lambda W: .5 * np.mean((np.maximum(Xw @ W.T, 0) @ aw - yw) ** 2)
res = np.maximum(Xw @ Ww.T, 0) @ aw - yw
gW = np.array([np.mean(res[:, None] * aw[j] * (Xw @ Ww[j] > 0)[:, None] * Xw, 0) for j in range(4)])
fdW = np.array([[(Lw(Ww + 1e-6 * np.eye(12)[3 * j + c].reshape(4, 3)) - Lw(Ww - 1e-6 * np.eye(12)[3 * j + c].reshape(4, 3))) / 2e-6 for c in range(3)] for j in range(4)])
check("notes Sec. 2: dL/dw_j = (1/n) sum (f - y) a_j 1{w_j.x > 0} x (away from kinks)", np.allclose(gW, fdW, atol=1e-6))
z = Rw.normal(size=50)
check("notes Sec. 4: a s(w.x) = a/2 s(w.x) + a/2 s(w.x)", np.allclose(1.7 * np.maximum(z, 0), .85 * np.maximum(z, 0) * 2))
dd = lambda f, x, d, a=1e-8: (f(x + a * d) - f(x)) / a
check("primer idea 1: one-sided slopes of |x| at 0 are |d|, of -|x| are -|d|",
      all(np.isclose(dd(abs, 0., d), abs(d)) and np.isclose(dd(lambda x: -abs(x), 0., d), -abs(d)) for d in (-2., -.5, .5, 2.)))
L1 = lambda a, w: .5 * (a * max(w, 0) - 1) ** 2
slopes = [(L1(1e-7 * np.cos(t), 1e-7 * np.sin(t)) - L1(0, 0)) / 1e-7 for t in np.linspace(0, 2 * np.pi, 73)]
check("primer idea 3: all one-sided slopes at (0,0) are 0; L(0.1, 0.1) = 0.49005 < 0.5",
      max(abs(v) for v in slopes) < 1e-6 and np.isclose(L1(.1, .1), .49005) and L1(0, 0) == .5)
check("primer idea 3: for w < 0 the loss is flat (dead neuron)", all(L1(a, w) == .5 for a in (-1, .3, 2) for w in (-.5, -.1)))
# idea 4: w.dL/dw = a dL/da, so ||w||^2 - a^2 is conserved by gradient flow (small-step GD)
aa, ww = .3, np.array([.2, -.4, .5]); x0 = Rw.normal(size=(5, 3)); y0 = Rw.normal(size=5)
c0 = ww @ ww - aa ** 2
for _ in range(20000):
    r_ = aa * np.maximum(x0 @ ww, 0) - y0
    ga = np.mean(r_ * np.maximum(x0 @ ww, 0)); gw = np.mean((r_ * aa * (x0 @ ww > 0))[:, None] * x0, 0)
    assert np.isclose(ww @ gw, aa * ga)
    aa, ww = aa - 1e-4 * ga, ww - 1e-4 * gw
check("primer idea 4: w.dL/dw = a dL/da and ||w||^2 - a^2 stays constant (step 1e-4)", abs((ww @ ww - aa ** 2) - c0) < 1e-4, f"drift {abs((ww @ ww - aa ** 2) - c0):.2e}")
nst = np.log(.1 / 5e-6) / np.log(1.001)
check("primer idea 5: ln(0.1/5e-6)/ln(1.001) ~ 9900 steps", abs(nst - 9900) < 50, f"{nst:.0f}")
done()
