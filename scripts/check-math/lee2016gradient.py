"""Lee et al. 2016: f(x,y) = x^2/2 + y^4/4 - y^2/2 (paper Sec. 3), stable set = x-axis."""
from common import *

print("lee2016gradient")
f = lambda p: .5 * p[0] ** 2 + .25 * p[1] ** 4 - .5 * p[1] ** 2
g = lambda p: np.array([p[0], p[1] ** 3 - p[1]])
check_grad("f", f, g, rand_pts(40, -1.6, 1.6))
check_hess_at("saddle (0,0)", f, [0, 0], [[1, 0], [0, -1]])
check_hess_at("min (0,1)", f, [0, 1], [[1, 0], [0, 2]])
L_region = max(1, 3 * 1.6 ** 2 - 1)
print(f"  gradient Lipschitz constant on |y|<=1.6: {L_region:.2f} -> alpha < {1 / L_region:.3f}")
lr = .1
check("chosen step is below 1/L on the plotted region", lr < 1 / L_region)
p = gd(g, [1.4, 0.0], lr, 3000)
check("start on the x-axis converges to the saddle", np.linalg.norm(p[-1]) < 1e-6)
ends = [gd(g, s, lr, 3000)[-1] for s in rand_pts(200, -1.5, 1.5, seed=5)]
n_sad = sum(np.linalg.norm(e) < 1e-3 for e in ends)
n_min = sum(min(np.linalg.norm(e - [0, 1]), np.linalg.norm(e + [0, 1])) < 1e-3 for e in ends)
check("200 random starts: none end at the saddle, all at minima", n_sad == 0 and n_min == 200, f"saddle {n_sad}, minima {n_min}")


def escape_iters(y0, lr=lr, x0=1.0):
    q = np.array([x0, y0])
    for k in range(200000):
        if abs(q[1]) > .5:
            return k
        q = q - lr * g(q)
    return None


ys = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 1e-12]
it = [escape_iters(y) for y in ys]
pred = [np.log(.5 / y) / np.log(1 + lr) for y in ys]
print("  escape iterations:", list(zip(ys, it)))
check("escape time grows like log(1/y0)/log(1+alpha)", all(abs(a - b) < .15 * b + 8 for a, b in zip(it, pred)))
check("escape time is unbounded as y0 -> 0", it[-1] > 5 * it[1])
# GD map Jacobian determinant
for a in [.1, .4]:
    for q in rand_pts(5, -1.5, 1.5, seed=7):
        J = np.array([[1 - a, 0], [0, (1 + a) - 3 * a * q[1] ** 2]])
        Jfd = np.column_stack([(q + e - a * g(q + e) - (q - e - a * g(q - e))) / 2e-6 for e in np.eye(2) * 1e-6])
        assert np.allclose(J, Jfd, atol=1e-5)
check("det Dg = (1-a)((1+a) - 3a y^2)", True)
yfold = lambda a: np.sqrt((1 + a) / (3 * a))
print(f"  fold line |y| = {yfold(.1):.2f} (a=0.1), {yfold(.4):.2f} (a=0.4)")
check("no fold inside |y|<=1.6 for a=0.1; fold appears for a=0.4", yfold(.1) > 1.6 and yfold(.4) < 1.6)

# in-page notes (papers/lee2016gradient/notes.md): worked example on f = x^2/2 - y^2/2, alpha = 0.2
q, traj = np.array([.40, .01]), []
for k in range(10):
    q = q - .2 * np.array([q[0], -q[1]])
    traj.append(q.copy())
print(f"  notes example: z1 = {traj[0]}, z2 = {traj[1]}, z10 = {traj[9]}")
check("notes example z1 = (0.320, 0.012), z2 = (0.256, 0.0144)", np.allclose(traj[0], [.32, .012]) and np.allclose(traj[1], [.256, .0144]))
check("notes example z10 ~ (0.043, 0.062)", np.allclose(traj[9], [.043, .062], atol=5e-4))
# g^{-1}(y) = argmin_x 1/2||x - y||^2 - a f(x) inverts g(x) = x - a grad f(x) when a < 1/L (paper's f, L = 2 on |y| <= 1)
a = .2
for q in rand_pts(5, -.9, .9, seed=3):
    y = q - a * g(q)
    xs = y.copy()
    for _ in range(200):          # stationarity x - y - a grad f(x) = 0, a contraction since a L < 1
        xs = y + a * g(xs)
    H = np.eye(2) - a * np.diag([1, 3 * xs[1] ** 2 - 1])
    assert np.all(np.linalg.eigvalsh(H) > 0) and np.allclose(xs, q, atol=1e-9), (q, xs)
check("proximal step on -f inverts the GD map (Prop. 8 construction)", True)
done()
