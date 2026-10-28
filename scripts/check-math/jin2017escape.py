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
done()
