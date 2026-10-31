"""Ahn, Magakyan, Cutkosky 2025: online-to-nonconvex conversion; schedule-free SGD is effective for nonconvex optimization.

SF-SGD: x_t = (1 - c_t) x_{t-1} + c_t z_t;  y_t = (1 - beta_t) z_t + beta_t x_t;  g_t = stoch. grad at y_t;  z_{t+1} = z_t - gamma g_t.
Definition 1: x is (lambda, eps)-stationary if inf_{E w = x} ||E grad F(w)|| + lambda E||w - x||^2 <= eps.
Main result (Sec. 5): with beta_t ~ U[beta, 1] (beta close to 1), c_t = 1 - beta and a large base step, SF-SGD finds a
(lambda, eps)-stationary point with O(lambda^{1/2} eps^{-7/2}) stochastic gradients, which is optimal (Zhang & Cutkosky 2024).
Proposition 1: L-smooth -> (L^2/eps, eps)-stationary gives ||grad F|| <= 2 eps  => O(eps^-4);
H-second-order-smooth -> (H/2, eps)-stationary gives ||grad F|| <= 2 eps  => O(eps^-3.5).
3D toy: nonsmooth, nonconvex f(x, y) = |x| + |y| + 0.5 sin(2x) sin(2y) with Gaussian gradient noise.
"""
from common import *

print("ahn2025general")
# notes' worked example: f = 1/2 (x - 2)^2, x0 = z1 = 0, gamma = 1/2, beta = 3/4, c = 1/2
g = lambda v: v - 2
x0, z = 0., 0.
x = .5 * x0 + .5 * z; y = .25 * z + .75 * x; z = z - .5 * g(y)
check("notes step 1: x1 = 0, y1 = 0, g1 = -2, z2 = 1", x == 0 and y == 0 and z == 1)
x = .5 * x + .5 * z; y = .25 * z + .75 * x; g2 = g(y); z = z - .5 * g2
check("notes step 2: x2 = 0.5, y2 = 0.625, g2 = -1.375, z3 = 1.6875", x == .5 and y == .625 and g2 == -1.375 and z == 1.6875)
# exponent bookkeeping of Proposition 1
check("smooth: lambda = L^2/eps -> lambda^(1/2) eps^(-7/2) = L eps^(-4)", -.5 - 3.5 == -4)
check("second-order smooth: lambda = H/2 (constant) -> eps^(-3.5)", -3.5 == -3.5)
# Goldstein intuition on |x|: gradients at points spread around the kink average to ~0 although each has norm 1
w = np.random.default_rng(0).normal(0, .05, 100000)
check("on |x| at the kink: mean of sign(w) for w spread around 0 is ~0, each |sign| = 1", abs(np.sign(w).mean()) < .01)

F = lambda v: abs(v[0]) + abs(v[1]) + .5 * np.sin(2 * v[0]) * np.sin(2 * v[1])
Gf = lambda v: np.array([np.sign(v[0]) + np.cos(2 * v[0]) * np.sin(2 * v[1]), np.sign(v[1]) + np.sin(2 * v[0]) * np.cos(2 * v[1])])
check_grad("nonsmooth toy away from the kinks", F, Gf, [p for p in rand_pts(20, -1.5, 1.5) if min(abs(p)) > .05][:6])
SIG = .5


GAM, BETA = .1, .9
r = np.random.default_rng(3); z = np.array([1.3, -1.1]); x = z.copy(); ys, gs, xs, zs = [], [], [], []
for t in range(1, 4001):
    x = BETA * x + (1 - BETA) * z            # c_t = 1 - beta
    bt = r.uniform(BETA, 1); y = (1 - bt) * z + bt * x
    gtrue = Gf(y); gt = gtrue + SIG * r.normal(size=2); z = z - GAM * gt
    ys.append(y.copy()); gs.append(gtrue); xs.append(x.copy()); zs.append(z.copy())
gs = np.array(gs); xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
late = slice(3000, 4000)
avg_norm = np.linalg.norm(gs[late].mean(0)); each = np.linalg.norm(gs[late], axis=1).mean()
spread = np.mean(np.sum((ys[late] - ys[late].mean(0)) ** 2, 1))
print(f"    SF-SGD (gamma {GAM}, beta {BETA}): late |mean of true gradients at y| = {avg_norm:.3f}, mean |gradient| = {each:.3f}, spread E|y - mean y|^2 = {spread:.4f}, |x_T| = {np.linalg.norm(xs[-1]):.3f}")
check("near the kink: individual gradients stay large but their local average is small (Definition 1 intuition)", avg_norm < .2 * each, f"{avg_norm:.3f} vs {each:.3f}")
sx, sz = np.linalg.norm(xs[late], axis=1).std(), np.linalg.norm(zs[late], axis=1).std()
check("x (EMA of z) is much steadier than the base iterate z", sx < .5 * sz, f"std |x| {sx:.3f} vs std |z| {sz:.3f}")
# in-page notes (papers/ahn2025general/notes.md): primer numbers
ws_, ps_ = np.array([-.01, .03]), np.array([.5, .5])
xm_ = ps_ @ ws_; gm_ = ps_ @ np.sign(ws_); sp_ = ps_ @ (ws_ - xm_) ** 2
check("primer 1: F = |x|, w in {-0.01, 0.03}: mean location 0.01, mean gradient 0, spread 0.0004", abs(xm_ - .01) < 1e-15 and gm_ == 0 and abs(sp_ - 4e-4) < 1e-15)
rng_ = np.random.default_rng(5); T_ = 10000; D_ = G_ = 1.
gs_ = rng_.choice([-1., 1.], size=T_); d_ = 0.; dl = []
for g in gs_:
    dl.append(d_); d_ = float(np.clip(d_ - D_ / (G_ * np.sqrt(T_)) * g, -D_, D_))
reg = max(np.dot(gs_, np.array(dl) - u) for u in (-D_, D_))
check("primer 2: online gradient descent regret <= D G sqrt(T) = 100 (random +-1 losses)", reg <= D_ * G_ * np.sqrt(T_), f"regret {reg:.1f}")
check("primer 3: 2^3.5 ~ 11.3; sqrt(L^2/eps) eps^-3.5 = L eps^-4", abs(2 ** 3.5 - 11.31) < 1e-2 and all(abs(np.sqrt(L_ ** 2 / e_) * e_ ** -3.5 - L_ * e_ ** -4) < 1e-6 * L_ * e_ ** -4 for L_ in (1, 3) for e_ in (.1, .01)))
save_json("ahn2025general", "data.json", {"gamma": GAM, "beta": BETA, "avg": round(float(avg_norm), 4), "each": round(float(each), 4), "spread": round(float(spread), 5)})
done()
