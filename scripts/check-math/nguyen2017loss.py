"""Nguyen & Hein 2017: wide layer (n_k >= N - 1) => regular critical points are global minima.

1) The notes' feature-rank example (3 samples, two tanh features + bias).
2) Lemma 3.5 mechanism for a linear readout: grad_v = F~^T r, so full row rank of F~ = [F, 1] forces r = 0 at a stationary readout.
3) Empirical width sweep (not the theorem): full-batch heavy-ball GD on a 1-hidden-layer tanh net, N = 8 points.
"""
from common import *

print("nguyen2017loss")
x = np.array([-1., 0, 1]); y = np.array([1., 0, 2])
F = np.column_stack([np.tanh(x), np.tanh(2 * x + 1), np.ones(3)])
dt = np.linalg.det(F); v = np.linalg.solve(F, y)
check("notes: det [F, 1] ~ -0.98225", abs(dt + .98225) < 1e-4, f"{dt:.5f}")
check("notes: readout (3.339, -2.326, 1.772) fits y exactly", np.allclose(v, [3.339, -2.326, 1.772], atol=1e-3) and np.allclose(F @ v, y))
# rank-deficient case: second feature collinear with the first and the bias -> stationary readout has residual
Fd = np.column_stack([np.tanh(x), .5 * np.tanh(x) + .2, np.ones(3)])
vd, *_ = np.linalg.lstsq(Fd, y, rcond=None); rd = Fd @ vd - y
check("rank-deficient features: least-squares readout is stationary (F^T r = 0) but r != 0", np.allclose(Fd.T @ rd, 0, atol=1e-10) and np.linalg.norm(rd) > .1, f"|r| = {np.linalg.norm(rd):.3f}")
# the set of (w2, c2) making [F,1] singular is thin: scan
W, Cc = np.meshgrid(np.linspace(-3, 3, 601), np.linspace(-3, 3, 601))
dets = np.tanh(-1) * (np.tanh(Cc) - np.tanh(W + Cc)) - np.tanh(-W + Cc) * (0 - np.tanh(1)) + (np.tanh(W + Cc) * 0 - np.tanh(Cc) * np.tanh(1))
# direct determinant of [[t(-1), t(-w+c), 1],[0, t(c), 1],[t(1), t(w+c), 1]]
t = np.tanh
D = t(-1) * (t(Cc) - t(W + Cc)) - t(-W + Cc) * (0 - t(1)) + 1 * (0 * t(W + Cc) - t(Cc) * t(1))
check("determinant formula used on the page matches numpy", abs(np.linalg.det(np.array([[t(-1), t(-.7 + .4), 1], [0, t(.4), 1], [t(1), t(.7 + .4), 1]])) - D[np.argmin(np.abs(Cc[:, 0] - .4)), np.argmin(np.abs(W[0] + .7))]) < 2e-3)
fr = [(np.abs(D) < e).mean() for e in (1e-2, 1e-3, 1e-4)]
check("near-singular area shrinks toward 0 as the tolerance shrinks (Lemma 4.4: measure zero)", fr[1] < .5 * fr[0] and fr[2] < .5 * fr[1], f"{fr[0]:.4f}, {fr[1]:.4f}, {fr[2]:.5f}")

# width sweep
N = 8
rng = np.random.default_rng(3)
X = np.linspace(-2, 2, N); Y = rng.uniform(-1, 1, N)
S = 6  # seeds


def train(n, steps=40000, lr=.05, beta=.9):
    r = np.random.default_rng(100 + n)
    w = r.normal(0, 1.5, (S, n)); c = r.normal(0, 1.5, (S, n)); vv = r.normal(0, .5, (S, n)) / np.sqrt(n); v0 = np.zeros(S)
    mw = np.zeros_like(w); mc = np.zeros_like(c); mv = np.zeros_like(vv); m0 = np.zeros(S)
    curve = []
    for k in range(steps):
        Z = w[:, :, None] * X[None, None, :] + c[:, :, None]  # S x n x N
        H = np.tanh(Z)
        out = (vv[:, :, None] * H).sum(1) + v0[:, None]
        res = out - Y[None, :]
        L = .5 * (res ** 2).mean(1)
        if k % 400 == 0:
            curve.append(L.copy())
        gout = res / N
        gv = (H * gout[:, None, :]).sum(2); g0 = gout.sum(1)
        gZ = vv[:, :, None] * (1 - H ** 2) * gout[:, None, :]
        gw = (gZ * X[None, None, :]).sum(2); gc = gZ.sum(2)
        mw = beta * mw - lr * gw; mc = beta * mc - lr * gc; mv = beta * mv - lr * gv; m0 = beta * m0 - lr * g0
        w += mw; c += mc; vv += mv; v0 += m0
    # gradient norm and feature rank at the end
    gn = np.sqrt((gw ** 2).sum(1) + (gc ** 2).sum(1) + (gv ** 2).sum(1) + g0 ** 2)
    rk = [int(np.linalg.matrix_rank(np.column_stack([H[s].T, np.ones(N)]), tol=1e-6)) for s in range(S)]
    xs = np.linspace(-2.3, 2.3, 93)
    best = int(np.argmin(L))
    pred = (vv[best][:, None] * np.tanh(w[best][:, None] * xs[None, :] + c[best][:, None])).sum(0) + v0[best]
    return L, np.array(curve), gn, rk, pred


rows, curves, preds = [], {}, {}
for n in range(1, 11):
    L, cu, gn, rk, pred = train(n)
    rows.append([n, sig(L), sig(gn, 3), rk])
    curves[n] = [sig(cu[:, s]) for s in range(S)]
    preds[n] = r(pred, 4)
    print(f"  width {n}: final loss min {L.min():.2e} median {np.median(L):.2e}; |grad| median {np.median(gn):.1e}; rank[F,1] {rk}")
mins = {rw[0]: min(rw[1]) for rw in rows}
check("width >= N-1 = 7: every seed reaches ~zero loss", all(max(rw[1]) < 1e-4 for rw in rows if rw[0] >= N - 1))
check("width 1-2: every seed stalls at a near-critical point with positive loss (bad critical points)", all(min(rw[1]) > 1e-3 and max(rw[2]) < 1e-3 for rw in rows if rw[0] <= 2))
check("N - 1 is sufficient, not necessary: width 3-6 also fit here", all(mins[n] < 1e-6 for n in range(3, 7)))
save_json("nguyen2017loss", "data.json", {"X": r(X), "Y": r(Y), "N": N, "rows": rows, "curves": curves, "xs": r(np.linspace(-2.3, 2.3, 93)), "preds": preds, "every": 400})

# in-page notes (papers/nguyen2017loss/notes.md): rounded matrix entries of Sec. 2 and the primer's examples
t = np.tanh
Fm = np.array([[t(-1), t(-1), 1], [0, t(1), 1], [t(1), t(3), 1]])
check("notes Sec. 2: entries -.762, .762, .995 (tanh(-1), tanh(1), tanh(3)); det ~ -0.982",
      np.allclose(np.round(Fm[:, :2], 3), [[-.762, -.762], [0, .762], [.762, .995]]) and round(np.linalg.det(Fm), 3) == -.982)
check("notes Sec. 2 box: grad_beta 1/2||A b - y||^2 = A^T (A b - y)",
      np.allclose(fd_grad(lambda b: .5 * np.sum((Fm @ b - np.array([1., 0, 2])) ** 2), np.array([.3, -.2, .5])), Fm.T @ (Fm @ np.array([.3, -.2, .5]) - np.array([1., 0, 2]))))
check("primer idea 1: 3x2 features without bias have rank 2 (< 3)", np.linalg.matrix_rank(Fm[:, :2]) == 2)
Rn = np.random.default_rng(53)
check("primer idea 3: random tanh feature matrices [F,1] (3 samples, 2 units) are full rank",
      all(abs(np.linalg.det(np.column_stack([t(np.outer([-1, 0, 1], w) + c), np.ones(3)]))) > 1e-9
          for w, c in zip(Rn.normal(size=(2000, 2)), Rn.normal(size=(2000, 2)))))
stat = lambda v: sorted(np.roots([1, 0, -v]).real) if v >= 0 else []
check("primer idea 4: u^2 - v = 0 has u = +-0.1 at v = 0.01 and no real root at v = -0.01",
      np.allclose(stat(.01), [-.1, .1]) and len(stat(-.01)) == 0 and np.all(np.iscomplex(np.roots([1, 0, .01]))))
check("primer idea 5: [1;2] has full column rank, [1 2] sends (2,-1) to 0",
      np.linalg.matrix_rank(np.array([[1], [2]])) == 1 and np.allclose(np.array([[1, 2]]) @ np.array([2, -1]), 0))
check("notes Sec. 7: widths [3,80,25,8,2] decrease after 80; [3,80,10,40,2] does not",
      all(a >= b for a, b in zip([80, 25, 8, 2], [25, 8, 2])) and not all(a >= b for a, b in zip([80, 10, 40, 2], [10, 40, 2])))
done()
