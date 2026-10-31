"""Li et al. 2018: a real 2-16-1 ReLU net; filter-normalized slices are invariant to the (cW1, cb1, W2/c) rescaling.

Writes papers/li2018visualizing/data.json (dataset + trained weights). The browser recomputes the slices live.
"""
from common import *

print("li2018visualizing")
rng = np.random.default_rng(7)
n = 80
t = rng.uniform(0, np.pi, n // 2)
X = np.concatenate([np.column_stack([np.cos(t), np.sin(t)]), np.column_stack([1 - np.cos(t), .5 - np.sin(t)])]) + rng.normal(0, .08, (n, 2))
X = (X - X.mean(0)) / X.std(0)
y = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])
Hd = 16


def forward(P, X):
    W1, b1, W2, b2 = P
    return np.maximum(X @ W1.T + b1, 0) @ W2 + b2


def loss(P):
    return .5 * np.mean((forward(P, X) - y) ** 2)


def grads(P):
    W1, b1, W2, b2 = P
    Z = X @ W1.T + b1
    A = np.maximum(Z, 0)
    e = (A @ W2 + b2 - y) / len(X)
    gW2 = A.T @ e
    gb2 = e.sum()
    dZ = np.outer(e, W2) * (Z > 0)
    return [dZ.T @ X, dZ.sum(0), gW2, gb2]


P = [rng.normal(0, np.sqrt(2 / 2), (Hd, 2)), np.zeros(Hd), rng.normal(0, np.sqrt(1 / Hd), Hd), 0.0]
m = [np.zeros_like(np.asarray(p, float)) for p in P]
v = [np.zeros_like(np.asarray(p, float)) for p in P]
for k in range(1, 6001):  # Adam
    G = grads(P)
    for i in range(4):
        m[i] = .9 * m[i] + .1 * G[i]
        v[i] = .999 * v[i] + .001 * np.square(G[i])
        P[i] = P[i] - .01 * (m[i] / (1 - .9 ** k)) / (np.sqrt(v[i] / (1 - .999 ** k)) + 1e-8)
acc = np.mean(np.sign(forward(P, X)) == y)
print(f"  trained: loss {loss(P):.4f}, train accuracy {acc:.3f}, |grad| {sum(np.sum(np.square(g)) for g in grads(P)) ** .5:.1e}")
check("network fits the data", acc > .97)


def directions(P, seed, normalize):
    r = np.random.default_rng(seed)
    W1, b1, W2, b2 = P
    dW1 = r.normal(size=W1.shape)
    dW2 = r.normal(size=W2.shape)
    if normalize:  # filter = incoming weights of one neuron
        dW1 = dW1 / np.linalg.norm(dW1, axis=1, keepdims=True) * np.linalg.norm(W1, axis=1, keepdims=True)
        dW2 = dW2 / np.linalg.norm(dW2) * np.linalg.norm(W2)
    return [dW1, np.zeros_like(b1), dW2, 0.0]  # biases left unperturbed


def slice_grid(P, normalize, ngrid=21, span=1.0):
    d, e = directions(P, 1, normalize), directions(P, 2, normalize)
    A = np.linspace(-span, span, ngrid)
    return np.array([[loss([P[i] + a * d[i] + b * e[i] for i in range(4)]) for a in A] for b in A])


c = 4.0
Pc = [c * P[0], c * P[1], P[2] / c, P[3]]
check("rescaled net computes the same function", np.allclose(forward(P, X), forward(Pc, X)))
g_n, g_nc = slice_grid(P, True), slice_grid(Pc, True)
g_u, g_uc = slice_grid(P, False), slice_grid(Pc, False)
check("filter-normalized slice is invariant to the rescaling", np.allclose(g_n, g_nc, atol=1e-10), f"max diff {np.max(np.abs(g_n - g_nc)):.1e}")
check("un-normalized slice changes with the rescaling", np.max(np.abs(g_u - g_uc)) > .05, f"max diff {np.max(np.abs(g_u - g_uc)):.3f}")
# hidden saddle example: L = 1/2 (x^2 + y^2 - 2 z^2)
Lh = lambda p: .5 * (p[0] ** 2 + p[1] ** 2 - 2 * p[2] ** 2)
check("hidden-saddle Hessian is diag(1,1,-2)", np.allclose(fd_hess(Lh, np.zeros(3)), np.diag([1, 1, -2]), atol=1e-4))
save_json("li2018visualizing", "data.json", {"X": r(X, 4), "y": y.tolist(), "W1": r(P[0], 6), "b1": r(P[1], 6), "W2": r(P[2], 6), "b2": float(P[3]),
                                             "train_loss": float(loss(P)), "train_acc": float(acc),
                                             "reported": {"depth": {"labels": ["ResNet-20", "ResNet-56", "ResNet-110"], "skip": [7.37, 5.89, 5.79], "noskip": [8.18, 13.31, 16.44]},
                                                          "width": {"labels": ["k=1", "k=2", "k=4", "k=8"], "skip": [5.89, 5.07, 4.34, 3.93], "noskip": [13.31, 10.26, 9.69, 8.70]}}})

# in-page notes (papers/li2018visualizing/notes.md): Sec. 2-4 numbers and the primer's numbers
yhat = lambda w1, w2, x=1.0: w2 * max(w1 * x, 0)
check("notes Sec. 2: (2,0.5) and (20,0.05) both predict 1; +0.1 gives 1.05 and 1.005",
      np.isclose(yhat(2, .5), 1) and np.isclose(yhat(20, .05), 1) and np.isclose(yhat(2.1, .5), 1.05) and np.isclose(yhat(20.1, .05), 1.005))
check("notes Sec. 3: 5% relative change: 2 -> 2.1 and 20 -> 21 both give 1.05", np.isclose(yhat(2 * 1.05, .5), 1.05) and np.isclose(yhat(20 * 1.05, .05), 1.05))
Lh = lambda p: .5 * (p[0] ** 2 + p[1] ** 2 - 2 * p[2] ** 2)
check_hess_at("notes Sec. 4: Hessian diag(1, 1, -2)", Lh, [0, 0, 0], np.diag([1, 1, -2]))
Rl2 = np.random.default_rng(62)
for D_, sd_ in [(10 ** 4, .01)]:
    cs = [(lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b))(Rl2.normal(size=D_), Rl2.normal(size=D_)) for _ in range(400)]
    check(f"primer idea 1: cosine of two Gaussian directions in D = {D_} has sd ~ 1/sqrt(D) = {sd_}", abs(np.std(cs) - sd_) < .0015, f"{np.std(cs):.4f}")
H3 = np.diag([1., 1, -2])
u1 = np.array([1, 0, 1]) / np.sqrt(2); u2 = np.array([1, 1, 0]) / np.sqrt(2)
check("primer idea 2: u^T H u = -0.5 for (1,0,1)/sqrt2 and +1 for (1,1,0)/sqrt2", np.isclose(u1 @ H3 @ u1, -.5) and np.isclose(u2 @ H3 @ u2, 1))
lam = np.r_[-2.0, np.ones(999)]
rq = [(lambda u: np.sum(lam * u * u) / (u @ u))(Rl2.normal(size=1000)) for _ in range(2000)]
check("primer idea 3: mean eigenvalue 0.997; random slices almost never show negative curvature",
      np.isclose(lam.mean(), .997) and min(rq) > 0 and abs(np.mean(rq) - .997) < .01, f"min {min(rq):.3f}")
check("primer idea 4: 10^12 float32 entries = 4 TB", 1e12 * 4 == 4e12)
A_ = Rl2.normal(size=(30, 30)); Hs = A_ + A_.T
gq = lambda th: Hs @ th   # gradient of th^T Hs th / 2
v_ = Rl2.normal(size=30); hv = (gq(np.ones(30) + 1e-5 * v_) - gq(np.ones(30) - 1e-5 * v_)) / 2e-5
check("primer idea 4: Hv from two gradients (central difference) equals H v", np.allclose(hv, Hs @ v_, atol=1e-6))
done()
