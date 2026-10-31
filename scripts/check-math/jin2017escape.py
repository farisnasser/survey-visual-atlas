"""Jin et al. 2017: perturbed GD on f = x^2/2 + cos(y) - 1; stuck band; dimension dependence."""
from common import *

print("jin2017escape")
f = lambda p: .5 * p[0] ** 2 + np.cos(p[1]) - 1
g = lambda p: np.array([p[0], -np.sin(p[1])])
check_grad("f", f, g, rand_pts(40, -4, 4))
check_hess_at("saddle", f, [0, 0], [[1, 0], [0, -1]])
check_hess_at("min (0,pi)", f, [0, np.pi], [[1, 0], [0, 1]])
check("global value -2 at (0, +-pi)", abs(f([0, np.pi]) + 2) < 1e-12)
lr = .2
p = gd(g, [1.5, 0.0], lr, 400)
check("plain GD from the stable set stays at the saddle", np.linalg.norm(p[-1]) < 1e-6)


def pgd(x0, r, g_thres, f_thres, t_thres, steps, rng, lr=lr):
    """Algorithm 2 of the paper (with its termination rule)."""
    x = np.array(x0, float)
    t_noise, x_tilde, kicks = -t_thres - 1, None, 0
    for t in range(steps):
        if np.linalg.norm(g(x)) <= g_thres and t - t_noise > t_thres:
            x_tilde, t_noise = x.copy(), t
            u = rng.normal(size=x.size)
            x = x_tilde + u * r * rng.uniform() ** (1 / x.size) / np.linalg.norm(u)
            kicks += 1
        if t - t_noise == t_thres and f(x) - f(x_tilde) > -f_thres:
            return x_tilde, kicks, t
        x = x - lr * g(x)
    return x, kicks, steps


rng = np.random.default_rng(1)
x, k, t_end = pgd([1.5, 0.0], r=.05, g_thres=1e-3, f_thres=.05, t_thres=30, steps=2000, rng=rng)
check("PGD: kick at the saddle escapes, kick at the minimum terminates", abs(f(x) + 2) < 1e-6 and k == 2, f"kicks {k}, f={f(x):.4f}, stopped at t={t_end}")
# stuck region: perturbed starts in a ball of radius r that do not decrease f by f_thres within t_thres steps
def stuck_fraction(gamma, t_thres, r=.5, n=40000, seed=0):
    fg = lambda p: np.array([p[0], -gamma * np.sin(p[1])])
    ff = lambda p: .5 * p[0] ** 2 + gamma * (np.cos(p[1]) - 1)
    rr = np.random.default_rng(seed)
    pts = rr.uniform(-r, r, (n, 2))
    pts = pts[np.linalg.norm(pts, axis=1) <= r]
    x = pts.copy()
    for _ in range(t_thres):
        x = x - lr * np.column_stack([x[:, 0], -gamma * np.sin(x[:, 1])])
    dec = ff(np.zeros(2)) - (.5 * x[:, 0] ** 2 + gamma * (np.cos(x[:, 1]) - 1))
    st = dec < .05 * gamma
    width = np.max(np.abs(pts[st, 1])) if st.any() else 0
    return st.mean(), width


for gm in [1, .5]:
    for tt in [10, 20, 40]:
        fr, w = stuck_fraction(gm, tt)
        print(f"  gamma={gm} t_thres={tt}: stuck fraction {fr:.4f}, band half-width {w:.4f}")
fr10, _ = stuck_fraction(1, 10)
fr40, _ = stuck_fraction(1, 40)
check("stuck band thins as t_thres grows", fr40 < fr10)
check("weaker curvature -> wider band", stuck_fraction(.5, 20)[0] > stuck_fraction(1, 20)[0])
# bent 3D variant: f = (x^2 + z^2)/2 + cos(y - 0.5 x^2) - 1
f3 = lambda p: .5 * (p[0] ** 2 + p[2] ** 2) + np.cos(p[1] - .5 * p[0] ** 2) - 1
g3 = lambda p: np.array([p[0] + p[0] * np.sin(p[1] - .5 * p[0] ** 2), -np.sin(p[1] - .5 * p[0] ** 2), p[2]])
check_grad("bent 3D", f3, g3, rand_pts(30, -1, 1, dim=3))
ev = eig(fd_hess(f3, np.zeros(3)))
check("bent 3D: strict saddle diag(1,-1,1)", np.allclose(ev, [-1, 1, 1], atol=1e-3))
rr = np.random.default_rng(3)
pts = rr.uniform(-.6, .6, (60000, 3))
pts = pts[np.linalg.norm(pts, axis=1) <= .6]
x = pts.copy()
for _ in range(20):
    x = x - lr * np.array([g3(q) for q in x]) if len(x) < 0 else x - lr * np.column_stack(
        [x[:, 0] + x[:, 0] * np.sin(x[:, 1] - .5 * x[:, 0] ** 2), -np.sin(x[:, 1] - .5 * x[:, 0] ** 2), x[:, 2]])
dec = -(.5 * (x[:, 0] ** 2 + x[:, 2] ** 2) + np.cos(x[:, 1] - .5 * x[:, 0] ** 2) - 1)
st = dec < .05
print(f"  bent pancake: stuck fraction {st.mean():.4f}")
corr = np.corrcoef(pts[st, 1], .5 * pts[st, 0] ** 2)[0, 1] if st.sum() > 10 else 0
check("bent pancake is thin and follows y ~ x^2/2", st.mean() < .2 and corr > .5, f"corr {corr:.2f}")
# escape iterations from the exact saddle vs dimension d (extra coordinates are stable, curvature 1)
def escape_d(d, seed):
    rr = np.random.default_rng(seed)
    x = np.zeros(d)
    u = rr.normal(size=d)
    x += .05 * rr.uniform() ** (1 / d) * u / np.linalg.norm(u)
    for t in range(5000):
        if np.cos(x[1]) - 1 < -1.0:
            return t
        gr = x.copy()
        gr[1] = -np.sin(x[1])
        x = x - lr * gr
    return 5000


for d in [2, 10, 100, 1000, 10000]:
    its = [escape_d(d, s) for s in range(40)]
    print(f"  d={d}: median escape iterations {np.median(its):.0f}")
m2 = np.median([escape_d(2, s) for s in range(40)])
m4 = np.median([escape_d(10000, s) for s in range(40)])
check("escape iterations grow only ~log d", m4 < 3 * m2, f"{m2:.0f} -> {m4:.0f}")

# in-page notes (papers/jin2017escape/notes.md): f = x^2/2 + cos y - 1, kick (0, 0.02), eta = 0.2
fJ = lambda p: .5 * p[0] ** 2 + np.cos(p[1]) - 1
gJ = lambda p: np.array([p[0], -np.sin(p[1])])
check_grad("notes: grad (x, -sin y)", fJ, gJ, rand_pts(10, -3, 3))
check_hess_at("notes: Hessian at (0,0) = diag(1, -1)", fJ, [0, 0], [[1, 0], [0, -1]])
check_hess_at("notes: Hessian diag(1, -cos y) at (0, 1.3)", fJ, [0, 1.3], [[1, 0], [0, -np.cos(1.3)]])
q1 = np.array([0, .02]) - .2 * gJ(np.array([0, .02]))
check("notes: y1 = 0.02 + 0.2 sin(0.02) ~ 0.0240", abs(q1[1] - .0240) < 5e-5 and abs(q1[1] - (.02 + .2 * np.sin(.02))) < 1e-15, f"{q1[1]:.6f}")
check("notes: x+ = 0.8x and y+ ~ 1.2y near 0", abs((np.array([.5, 0]) - .2 * gJ(np.array([.5, 0])))[0] - .4) < 1e-15 and abs(q1[1] / .02 - 1.2) < 1e-3)
ys = np.linspace(-10, 10, 20001)
check("notes: l = rho = 1 (|cos y| <= 1, |sin y| <= 1) and f >= -2 (bounded below)", np.max(np.abs(np.cos(ys))) <= 1 and np.max(np.abs(np.sin(ys))) <= 1 and min(np.cos(ys) - 1) >= -2)
# primer
check("primer: 2 l Delta / eps^2 = 20000 for l = 1, Delta = 1, eps = 0.01", abs(2 * 1 * 1 / .01 ** 2 - 20000) < 1e-6)
# descent lemma on the notes' f (l = 1, eta = 1/l)
check("primer: f(x - grad/l) <= f(x) - |grad|^2/(2l) on the notes' f", all(fJ(q - gJ(q)) <= fJ(q) - gJ(q) @ gJ(q) / 2 + 1e-12 for q in rand_pts(200, -4, 4)))
check("primer: gamma^3/rho^2 = eps gamma/rho at gamma = sqrt(rho eps); rho = 1, eps = 1e-4 -> 0.01",
      all(abs(np.sqrt(r * e) ** 3 / r ** 2 - e * np.sqrt(r * e) / r) < 1e-15 for r, e in [(1, 1e-4), (3, .02)]) and abs(np.sqrt(1e-4) - .01) < 1e-15)
w = .05
disk = 2 / np.pi * (w * np.sqrt(1 - w * w) + np.arcsin(w)); ball3 = 1.5 * w - .5 * w ** 3
U2 = np.random.default_rng(0).uniform(-1, 1, (2000000, 2)); U2 = U2[(U2 ** 2).sum(1) <= 1]
U3 = np.random.default_rng(1).uniform(-1, 1, (2000000, 3)); U3 = U3[(U3 ** 2).sum(1) <= 1]
check("primer: band |y| <= 0.05 holds ~0.064 of the disk (Monte Carlo)", abs(disk - .064) < 5e-4 and abs(np.mean(np.abs(U2[:, 1]) <= w) - disk) < 1.5e-3, f"{disk:.5f}")
check("primer: slab |z| <= 0.05 holds ~0.075 of the 3-ball (Monte Carlo)", abs(ball3 - .075) < 5e-4 and abs(np.mean(np.abs(U3[:, 2]) <= w) - ball3) < 1.5e-3, f"{ball3:.5f}")
check("primer: k = log(1/delta)/log(1/p) = 20 for p = 1/2, delta = 1e-6", int(np.ceil(np.log(1e6) / np.log(2))) == 20 and .5 ** 20 < 1e-6 < .5 ** 19)
done()
