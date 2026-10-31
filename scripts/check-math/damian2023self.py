"""Damian, Nichani, Lee 2023: self-stabilization at the edge of stability.

App. B toy, restricted to (x, y): L(x, y) = (2/eta + b y) x^2 / 2 - (alpha / b) y   (the paper's extra -z term only moves
the constrained trajectory along z and does not touch x, y).  Sharpness along u = e_x is S = 2/eta + b y, grad S = (0, b),
so beta = |grad S|^2 = b^2 and alpha = -grad L . grad S.  GD on this L is exactly the simplified dynamics Eq. (3):
    x_{t+1} = -(1 + eta y'_t) x_t,   y'_{t+1} = y'_t + eta (alpha - beta x_t^2 / 2),   y' = b y = S - 2/eta.
ODE Eq. (4): X' = X Y, Y' = alpha - beta X^2 / 2; Lemma 3: g = h(beta X^2 / (2 alpha)) + Y^2 / alpha conserved, h(z) = z - log z - 1.
"""
from common import *

print("damian2023self")
ETA, B, AL = 1.0, 1.0, .002         # eta = 1 so 2/eta = 2; beta = B^2 = 1
BE = B * B
L = lambda z: (2 / ETA + B * z[1]) * z[0] ** 2 / 2 - AL / B * z[1]
G = lambda z: np.array([(2 / ETA + B * z[1]) * z[0], B * z[0] ** 2 / 2 - AL / B])
check_grad("App. B toy", L, G, rand_pts(6, -.5, .5))
H = fd_hess(L, [.1, .2])
check("sharpness along u = e_x is 2/eta + b y at x = 0", abs(fd_hess(L, [0, .2])[0, 0] - (2 / ETA + B * .2)) < 1e-6)
check("alpha = -grad L . grad S on the x = 0 line", abs(-(G([0, .1]) @ np.array([0, B])) - AL) < 1e-12)


def gd(z0, T):
    z = np.array(z0, float); out = [z.copy()]
    for _ in range(T):
        z = z - ETA * G(z); out.append(z.copy())
    return np.array(out)


X0 = .6 * np.sqrt(2 * AL / BE)          # initial oscillation 0.6 delta
T = 450                                 # ~4 cycles shown on the page
tr = gd([X0, 0], T)
x, yp = tr[:, 0], B * tr[:, 1]
pred_x = -(1 + ETA * yp[:-1]) * x[:-1]
pred_y = yp[:-1] + ETA * (AL - BE * x[:-1] ** 2 / 2)
check("GD on the toy is exactly Eq. (3)", np.abs(pred_x - x[1:]).max() < 1e-14 and np.abs(pred_y - yp[1:]).max() < 1e-13)
check("x flips sign every step (oscillation along u)", np.all(np.sign(x[1:200]) == -np.sign(x[:199])))


def gfun(X, Y):
    z = BE * X * X / (2 * AL)
    return z - np.log(z) - 1 + Y * Y / AL


# Lemma 3 for the ODE (RK4)
def rk4(X, Y, dt, n):
    f = lambda X, Y: (X * Y, AL - BE * X * X / 2)
    out = [(X, Y)]
    for _ in range(n):
        k1 = f(X, Y); k2 = f(X + dt / 2 * k1[0], Y + dt / 2 * k1[1]); k3 = f(X + dt / 2 * k2[0], Y + dt / 2 * k2[1]); k4 = f(X + dt * k3[0], Y + dt * k3[1])
        X += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); Y += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); out.append((X, Y))
    return np.array(out)


O = rk4(X0, 0., .01, 400000)
gv = gfun(O[:, 0], O[:, 1])
check("Lemma 3: g(X, Y) conserved along the ODE", np.abs(gv - gv[0]).max() < 1e-6 * gv[0], f"g = {gv[0]:.4f}, drift {np.abs(gv - gv[0]).max():.1e}")
delta = np.sqrt(2 * AL / BE)
check("fixed point (delta, 0), delta = sqrt(2 alpha / beta)", abs(AL - BE * delta ** 2 / 2) < 1e-15, f"delta = {delta:.4f}")
check("min X along the orbit = X(0) when Y(0) = 0", abs(O[:, 0].min() - X0) < 1e-4)
# one full period: time averages of Y and X^2
cross = np.where((O[:-1, 1] < 0) & (O[1:, 1] >= 0))[0]
a, b_ = cross[0], cross[1]
check("time average of Y over a cycle is 0 (mean sharpness = 2/eta)", abs(O[a:b_, 1].mean()) < 1e-3 * np.abs(O[:, 1]).max(), f"mean Y = {O[a:b_, 1].mean():.1e}")
check("time average of X^2 over a cycle is 2 alpha / beta", abs((O[a:b_, 0] ** 2).mean() - delta ** 2) / delta ** 2 < 1e-2, f"{(O[a:b_, 0] ** 2).mean():.4f} vs {delta ** 2:.4f}")
# discrete GD: g nearly conserved, mean sharpness ~ 2/eta
gg = gfun(np.abs(x[1:]), yp[1:])
ode_period = (b_ - a) * .01
check("first GD cycle length matches the ODE period (time unit = 1 step)", abs(np.diff(np.where((yp[:-1] < 0) & (yp[1:] >= 0))[0])[0] - ode_period) / ode_period < .1, f"ODE period {ode_period:.1f}")
check("GD over 450 steps: g grows (the discrete map spirals outward)", gg[-1] > gg[0] and gg.max() < 6 * gg[0], f"g {gg[0]:.3f} -> max {gg.max():.3f}")
J = lambda X, Y: np.array([[-(1 + ETA * Y), -ETA * X], [-ETA * BE * X, 1]])     # Jacobian of Eq. (3) in (x, y')
ev = np.linalg.eigvals(J(-delta, 0) @ J(delta, 0))
check("two-step Jacobian at the period-2 orbit (+-delta, 0): eigenvalues (1 - d^2) +- 2i d, modulus > 1", np.allclose(sorted(ev.imag), [-2 * delta, 2 * delta]) and abs(abs(ev[0]) - np.sqrt((1 - delta ** 2) ** 2 + 4 * delta ** 2)) < 1e-12 and abs(ev[0]) > 1, f"|ev| = {abs(ev[0]):.5f}, 1 + delta^2 = {1 + delta ** 2:.5f}")
long = gd([X0, 0], 6000)
check("left running much longer, the discrete map leaves the EoS regime", not np.all(np.isfinite(long)) or np.abs(long[:, 0]).max() > 1, "Theorem 1 only covers T = O(1/eps) steps")
S = 2 / ETA + yp
check("GD: time-averaged sharpness = 2/eta within 1%", abs(S[1:].mean() - 2 / ETA) < .01, f"mean S = {S[1:].mean():.4f}, range [{S.min():.3f}, {S.max():.3f}]")
period = np.diff(np.where((yp[:-1] < 0) & (yp[1:] >= 0))[0])
print(f"    GD cycle lengths: {period[:6].tolist()} steps; max |x| = {np.abs(x).max():.3f}, delta = {delta:.3f}")
# smaller x0 -> larger cycles (Corollary 1)
amp = {x0: np.abs(gd([x0, 0], 300)[:, 1]).max() * B for x0 in [.05, .02, .005]}
check("smaller initial oscillation -> larger sharpness swings (Cor. 1)", amp[.05] < amp[.02] < amp[.005], f"max |S - 2/eta|: {amp}")
# in-page notes (papers/damian2023self/notes.md): Sec. 2 threshold |1 - eta lambda| < 1 <=> 0 < eta lambda < 2
check("notes: |1 - eta lambda| < 1 exactly when 0 < eta lambda < 2", all((abs(1 - q) < 1) == (0 < q < 2) for q in np.linspace(-1, 3, 4001)))
# Sec. 3: d/dtheta of x^2/2 S(theta) = x^2/2 grad S (toy S(y) = 2 + y), GD drift -eta x^2/2 grad S
Lt = lambda p: .5 * (2 + p[1]) * p[0] ** 2
check_grad("notes/primer: toy L = (2 + y) x^2 / 2, dL/dy = x^2/2", Lt, lambda p: np.array([(2 + p[1]) * p[0], .5 * p[0] ** 2]), rand_pts(10, -1, 1))
check("primer: eta = 1, x = 1 moves y by -0.5", np.isclose(-1 * .5 * 1 ** 2, -.5))
# primer: zig-zag multipliers
seq = lambda lam, n=3: [.1 * (1 - lam) ** k for k in range(n)]
check("primer: lambda = 1.8: 0.1, -0.08, 0.064; lambda = 2.2: 0.1, -0.12, 0.144", np.allclose(seq(1.8), [.1, -.08, .064]) and np.allclose(seq(2.2), [.1, -.12, .144]))
# primer: Eq. (3) with eta = 1, alpha = 0.01, beta = 1
ea, al, be = 1., .01, 1.
step = lambda x, y: (-(1 + ea * y) * x, y + ea * (al - be * x * x / 2))
check("primer: Eq. 3: x = 0 -> y += 0.01; |x| = 0.5 -> y += -0.115; threshold sqrt(2 alpha/beta) ~ 0.14",
      np.isclose(step(0, 0)[1], .01) and np.isclose(step(.5, 0)[1], -.115) and np.isclose(np.sqrt(2 * al / be), .14, atol=.005))
check("primer: y > 0 grows |x|, y < 0 shrinks it", abs(step(.1, .2)[0]) > .1 and abs(step(.1, -.2)[0]) < .1)
# one full cycle of the four stages from a tiny x, then the outward drift of the discrete map
x, y, ys, xs_ = 1e-3, -.05, [], []
for t in range(150):
    x, y = step(x, y); ys.append(y); xs_.append(abs(x))
ys, xs_ = np.array(ys), np.array(xs_)
cr = np.where(np.diff(np.sign(ys)) != 0)[0]                 # sign changes of y = S - 2/eta
c1 = xs_[cr[0]:cr[2] + 1]
check("primer: one loop: y crosses 0 upward, |x| peaks above sqrt(2 alpha/beta), y crosses back, |x| shrinks, y rises again",
      len(cr) >= 3 and ys[cr[0] + 1] > 0 and c1.max() > np.sqrt(2 * al / be) and ys[cr[1] + 1] < 0 and xs_[cr[2]] < .1 * c1.max() and ys[cr[2] + 1] > 0,
      f"crossings {cr[:3]}, peak |x| {c1.max():.3f}, |x| at the restart {xs_[cr[2]]:.2e}")
x, y, pk = 1e-3, -.05, []
for t in range(1000):
    x, y = step(x, y); pk.append(abs(x))
    if abs(x) > 10: break
check("primer: in the two-number map the cycles grow (not a closed orbit)", max(pk) > 10 * c1.max(), f"|x| exceeds 10 at step {t}")
done()
