"""Zhang, Saxe, Latham 2026: saddle-to-saddle on the invariant manifold; linear vs quadratic networks."""
from common import *

print("zhang2026saddle")
s1, s2 = 2.0, 0.6
L = lambda a: .5 * (s1 - a[0] ** 2) ** 2 + .5 * (s2 - a[1] ** 2) ** 2
Lg = lambda a: np.array([-2 * a[0] * (s1 - a[0] ** 2), -2 * a[1] * (s2 - a[1] ** 2)])
check_grad("manifold loss", L, Lg, rand_pts(40, -1.7, 1.7))
check_hess_at("origin", L, [0, 0], [[-2 * s1, 0], [0, -2 * s2]])
H = check_hess_at("(sqrt s1, 0)", L, [np.sqrt(s1), 0], [[4 * s1, 0], [0, -2 * s2]])
check("(sqrt s1,0) is a strict saddle with weak negative curvature -2 s2", eig(H)[0] < 0)
check("(sqrt s1, sqrt s2) is the global minimum", L([np.sqrt(s1), np.sqrt(s2)]) < 1e-20)
# the manifold loss is the exact loss of a 2-layer linear net W2 W1 with whitened inputs,
# target diag(s1,s2) and balanced, aligned weights W1 = W2 = diag(a1, a2) (scaled by 1/sqrt2 per factor here: W = diag(a^2))
def net_loss(W1, W2):
    return .5 * np.sum((np.diag([s1, s2]) - W2 @ W1) ** 2)
check("manifold loss = linear-net loss at W1=W2=diag(a)", all(abs(L(a) - net_loss(np.diag(a), np.diag(a))) < 1e-12 for a in rand_pts(20, -1.5, 1.5)))
# the full linear net started on the manifold stays on it (invariance) and passes two saddles
def full_gd(W1, W2, lr, steps):
    T = np.diag([s1, s2])
    hist = []
    for _ in range(steps):
        E = T - W2 @ W1
        W1, W2 = W1 + lr * W2.T @ E, W2 + lr * E @ W1.T
        hist.append(.5 * np.sum(E ** 2))
    return W1, W2, np.array(hist)
a0 = 1e-3
W1, W2, h = full_gd(np.diag([a0, a0]), np.diag([a0, a0]), .02, 2500)
check("aligned balanced init stays diagonal (invariant manifold)", abs(W1[0, 1]) + abs(W1[1, 0]) < 1e-12)
p = gd(Lg, [a0, a0], .02, 2500)
check("manifold GD = full-net GD on the manifold", np.allclose(np.diag(W1) ** 2, p[-1] ** 2, atol=1e-8))
# two plateaus: loss near L(0)=(s1^2+s2^2)/2, then near s2^2/2, then 0
lv = [L(q) for q in p]
t1 = int(np.argmax(np.array(lv) < .5 * s2 ** 2 + .05))
t2 = int(np.argmax(np.array(lv) < 1e-3))
print(f"  plateau 1 ends ~step {t1}, plateau 2 ends ~step {t2}")
check("second plateau sits at the width-1 saddle value s2^2/2", 0 < t1 < t2 and t2 - t1 > 100)
# random small init, full non-aligned linear net: still saddle-to-saddle (low-rank first)
rng = np.random.default_rng(0)
Wa, Wb, hh = full_gd(rng.normal(0, 1e-3, (2, 2)), rng.normal(0, 1e-3, (2, 2)), .02, 3000)
check("random small init: two drops (rank 1 then rank 2)", hh[-1] < 1e-6 and np.any(np.abs(hh - .5 * s2 ** 2) < .01))
# quadratic-in-weights net f(x) = sum_i v_i (u_i^T x)^2 : rich-get-richer between units
def quad_run(H=6, sig=1e-2, steps=6000, lr=.05, seed=1):
    r = np.random.default_rng(seed)
    X = r.normal(size=(200, 2))
    y = (X[:, 0] ** 2) - .6 * (X[:, 1] ** 2)       # teacher: two units
    u = r.normal(0, sig, (H, 2))
    v = r.normal(0, sig, H)
    norms, loss = [], []
    for k in range(steps):
        z = X @ u.T
        pred = (z ** 2) @ v
        e = pred - y
        loss.append(.5 * np.mean(e ** 2))
        gv = (z ** 2).T @ e / len(X)
        gu = 2 * ((e[:, None] * z * v[None, :]).T @ X) / len(X)
        v, u = v - lr * gv, u - lr * gu
        if k % 20 == 0:
            norms.append(np.sqrt(v ** 2 + np.sum(u ** 2, 1)))
    return np.array(loss), np.array(norms)
ql, qn = quad_run()
order = np.argsort(-qn[-1])
first = int(np.argmax(qn[:, order[0]] > .3))
second = int(np.argmax(qn[:, order[1]] > .3))
print(f"  quadratic net: final loss {ql[-1]:.2e}; unit A grows at snapshot {first}, unit B at {second}; others final norm {np.round(np.sort(qn[-1])[:-2], 3)}")
check("quadratic net: units grow one at a time; rest stay small", second > first + 5 and np.sort(qn[-1])[-3] < .2)
done()
