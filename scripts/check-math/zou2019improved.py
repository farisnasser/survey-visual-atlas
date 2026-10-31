"""Zou & Gu 2019: GD on over-parameterized ReLU networks — the two proof ingredients on a real small problem.

Data: n = 16 points on the unit sphere in R^5 with last coordinate mu = 0.5 (Assumption 3.1), min pairwise distance phi > 0
(Assumption 3.2), random labels.  Network (one hidden layer, the paper's L = 1 case): f(x) = sum_r v_r relu(w_r . x),
W rows ~ N(0, 2/m I) and output weights v ~ N(0, 1/d) as in Algorithm 1 (v fixed: the paper trains the hidden layers), square loss
L(W) = 1/2 sum_i (f(x_i) - y_i)^2, GD with step eta ~ 1/n (width-independent in this scaling).
Step eta = 1/lambda_max of the Gram matrix H_ij = sum_r v_r^2 1[w_r.x_i > 0] 1[w_r.x_j > 0] x_i.x_j at init, which is
Theta(1/m) as in Theorem 3.3 (eta = O(k/(L^2 m))).
Shown: (i) linear convergence of the training loss (gradient lower bound); (ii) the trajectory stays close to init: the
spectral-norm movement |W_T - W_0|_2 (relative to |W_0|_2) and the fraction of ReLU activation patterns that flip both shrink
with width (the trajectory-length / semi-smoothness ingredients of the proof).
"""
from common import *

print("zou2019improved")
n, d, mu = 10, 10, .5
R0 = np.random.default_rng(86)
Z = R0.normal(size=(n, d - 1)); Z = Z / np.linalg.norm(Z, axis=1, keepdims=True) * np.sqrt(1 - mu * mu)
X = np.c_[Z, np.full(n, mu)]
y = R0.choice([-1., 1.], n)
phi = min(np.linalg.norm(X[i] - X[j]) for i in range(n) for j in range(i))
check("Assumption 3.1: |x_i| = 1 and (x_i)_d = mu", np.allclose(np.linalg.norm(X, axis=1), 1) and np.allclose(X[:, -1], mu))
check("Assumption 3.2: phi = min |x_i - x_j| > 0", phi > 0, f"phi = {phi:.3f}")
relu = lambda z: np.maximum(z, 0)


def init(m, seed):
    R = np.random.default_rng(seed)
    return R.normal(0, np.sqrt(2 / m), (m, d)), R.normal(0, np.sqrt(1 / d), m)


def loss(W, v):
    return .5 * np.sum((relu(X @ W.T) @ v - y) ** 2)


def grad(W, v):
    Zp = X @ W.T; r = relu(Zp) @ v - y
    return ((r[:, None] * (Zp > 0)) * v[None, :]).T @ X


W0, v0 = init(30, 1)
check_grad("loss in W (m = 30)", lambda w: loss(w.reshape(30, d), v0), lambda w: grad(w.reshape(30, d), v0).ravel(), [W0.ravel(), W0.ravel() * .5])


def gram(W, v):
    S = (X @ W.T > 0).astype(float)
    return (S * v ** 2) @ S.T * (X @ X.T)


def train(m, seed=0, T=3000):
    W, v = init(m, seed); W0 = W.copy()
    eta = 1 / np.linalg.eigvalsh(gram(W, v))[-1]
    L = [loss(W, v)]
    for t in range(T):
        W = W - eta * grad(W, v); L.append(loss(W, v))
    mv = np.linalg.norm(W - W0, 2) / np.linalg.norm(W0, 2)
    flips = np.mean((X @ W.T > 0) != (X @ W0.T > 0))
    return np.array(L), mv, flips, W0, W, v, eta


widths = [32, 128, 512, 2048, 8192]
out = {"widths": widths, "phi": round(float(phi), 4)}
moves = {}
for m in widths:
    L, mv, fl, W0_, WT, v, eta = train(m)
    k = int(np.argmax(L < 1e-6 * L[0])) if np.any(L < 1e-6 * L[0]) else -1
    moves[m] = (mv, fl)
    print(f"  m = {m:5d}: eta = {eta:.3f}, L_0 = {L[0]:.3f}, L_T = {L[-1]:.2e}, iters to 1e-6 L_0: {k}, relative spectral move {mv:.4f}, flipped patterns {fl:.4f}")
    out[f"loss{m}"] = sig(np.maximum(L[::10], 1e-14), 3); out[f"mv{m}"] = round(float(mv), 5); out[f"fl{m}"] = round(float(fl), 5); out[f"k{m}"] = k
check("wide nets (m >= 2048) converge linearly to 1e-6 of the initial loss", all(out[f"k{m}"] > 0 for m in (2048, 8192)))
check("relative spectral-norm movement |W_T - W_0|_2 / |W_0|_2 shrinks with width", moves[8192][0] < moves[2048][0] < moves[512][0] < moves[128][0])
check("fraction of flipped activation patterns shrinks with width", moves[8192][1] < moves[512][1] < moves[32][1])
# 2-D slice of the training loss around init for m = 32 and m = 2048: W0 + a D1 + b D2, D1 = W_T - W_0, D2 random, same norm
for m in (32, 8192):
    L, mv, fl, W0_, WT, v, eta = train(m)
    D1 = WT - W0_; R = np.random.default_rng(7); D2 = R.normal(size=D1.shape); D2 -= (D2 * D1).sum() / (D1 * D1).sum() * D1; D2 *= np.linalg.norm(D1) / np.linalg.norm(D2)
    A = np.linspace(-.5, 1.5, 41); Bg = np.linspace(-1, 1, 41)
    grid = [[loss(W0_ + a * D1 + b * D2, v) for a in A] for b in Bg]
    out[f"slice{m}"] = [sig(row, 4) for row in grid]
    check(f"m = {m}: slice value at (1, 0) equals the final training loss", abs(grid[20][30] - L[-1]) < 1e-9 + 1e-6 * L[0])
    # is the slice convex along the training direction (b = 0)?
    row = np.array(grid[20]); sec = row[2:] - 2 * row[1:-1] + row[:-2]
    out[f"convex{m}"] = bool(np.all(sec > -1e-9))
    print(f"  m = {m}: slice along the training direction convex: {out[f'convex{m}']}, min second difference {sec.min():.2e}")
save_json("zou2019improved", "data.json", out)
# in-page notes (papers/zou2019improved/notes.md): telescoping with L_t = q^t L0 and steps C(sqrt L_t - sqrt L_{t+1})
for q_, C_, L0_, T_ in [(.9, 1., 1., 50), (.5, 3., 2., 10), (.99, .2, 5., 2000)]:
    Lt = L0_ * q_ ** np.arange(T_ + 1)
    path = np.sum(C_ * (np.sqrt(Lt[:-1]) - np.sqrt(Lt[1:])))
    assert np.isclose(path, C_ * (np.sqrt(L0_) - np.sqrt(Lt[-1]))) and path <= C_ * np.sqrt(L0_)
check("notes: telescoped path length = C(sqrt L0 - sqrt LT) <= C sqrt L0 for any T", True)
# primer idea 1: factor 0.99 per step -> ln(1e6)/(-ln 0.99) ~ 1375 steps; eta*mu = (k/(L^2 m)) (m phi/(k n^2)) = phi/(n^2 L^2)
check("primer: 0.99^t <= 1e-6 first at t = 1375", int(np.ceil(np.log(1e6) / -np.log(.99))) == 1375 and .99 ** 1375 <= 1e-6 < .99 ** 1374)
for k_, L_, m_, ph, n_ in [(1, 3, 1e6, .1, 50), (4, 10, 1e9, .5, 1000)]:
    assert np.isclose((k_ / (L_ ** 2 * m_)) * (m_ * ph / (k_ * n_ ** 2)), ph / (n_ ** 2 * L_ ** 2))
check("primer: eta*mu ~ phi/(n^2 L^2), independent of the width m", True)
# primer idea 2: movement sqrt(k n^2 log n/(m phi)) halves when m quadruples; <= tau = phi^1.5 n^-3 L^-6 iff m >= k n^8 L^12/phi^4 (logs dropped)
mv = lambda m_, k_=1, n_=100, ph=.3: np.sqrt(k_ * n_ ** 2 * np.log(n_) / (m_ * ph))
check("primer: quadrupling m halves the movement bound", np.isclose(mv(4e6) / mv(1e6), .5))
k_, n_, L_, ph = 2., 30., 3., .4
mthr = k_ * n_ ** 8 * L_ ** 12 / ph ** 4
mvn = lambda m_: np.sqrt(k_ * n_ ** 2 / (m_ * ph)); tau_ = ph ** 1.5 * n_ ** -3 * L_ ** -6
check("primer: movement <= tau exactly when m >= k n^8 L^12/phi^4", np.isclose(mvn(mthr), tau_) and mvn(2 * mthr) < tau_ < mvn(mthr / 2))
# primer idea 3: duplicated input with labels +1, -1: (1/2)((u-1)^2 + (u+1)^2) = u^2 + 1 >= 1
uu = np.linspace(-3, 3, 6001)
check("primer: duplicate input with opposite labels has loss >= 1, minimum at u = 0", np.isclose((.5 * ((uu - 1) ** 2 + (uu + 1) ** 2)).min(), 1) and np.isclose(uu[np.argmin((uu - 1) ** 2 + (uu + 1) ** 2)], 0))
# primer idea 4: n = 1000: n^8 = 1e24, n^24 = 1e72, ratio 1e48
check("primer: 1000^8 = 1e24, 1000^24 = 1e72, n^16 = 1e48", 1000 ** 8 == 10 ** 24 and 1000 ** 24 == 10 ** 72 and 1000 ** 16 == 10 ** 48)
done()
