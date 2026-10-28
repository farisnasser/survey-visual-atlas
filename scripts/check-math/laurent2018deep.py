"""Laurent & von Brecht 2018: convex differentiable losses vs the nondifferentiable counterexample."""
from common import *

print("laurent2018deep")
T = 2.0
sq = lambda p: (p[0] * p[1] - T) ** 2
sq_g = lambda p: np.array([2 * (p[0] * p[1] - T) * p[1], 2 * (p[0] * p[1] - T) * p[0]])
lc = lambda p: np.log(np.cosh(p[0] * p[1] - T))
lc_g = lambda p: np.array([np.tanh(p[0] * p[1] - T) * p[1], np.tanh(p[0] * p[1] - T) * p[0]])
P = rand_pts(40, -1.8, 1.8)
check_grad("squared", sq, sq_g, P)
check_grad("log-cosh", lc, lc_g, P)
for nm, f in [("squared", sq), ("log-cosh", lc)]:
    ev = eig(fd_hess(f, [0, 0]))
    check(f"{nm}: origin is a strict saddle", ev[0] < 0 < ev[1], str(np.round(ev, 3)))
for nm, f, g in [("squared", sq, sq_g), ("log-cosh", lc, lc_g)]:
    ends = [f(gd(g, s, .02, 8000, clip=1.8)[-1]) for s in rand_pts(40, -1.7, 1.7, seed=2)]
    check(f"{nm}: random starts reach zero loss", max(ends) < 1e-4, f"max {max(ends):.1e}")
# counterexample (Eq. 22-23): f(x,y)=|x|+(1-y)_+ - 1 ; slice W1=alpha, W2=(1-beta, beta)
F = lambda p: abs(p[0] * (1 - p[1])) + max(1 - p[0] * p[1], 0) - 1


def F_sub(p):
    a, b = p
    s = np.sign(a * (1 - b))
    act = 1.0 if 1 - a * b > 0 else 0.0
    return np.array([s * (1 - b) - b * act, -s * a - a * act])


f_paper = lambda W1, W2: abs((W2 * W1)[0]) + max(1 - (W2 * W1)[1], 0) - 1
check("slice equals the paper's F(W1,W2)", all(abs(F(p) - f_paper(p[0], np.array([1 - p[1], p[1]]))) < 1e-12 for p in P))
check("F(0,0)=0 and global value -1", F([0, 0]) == 0 and abs(F([1.5, 1.0]) + 1) < 1e-12)
nb = np.random.default_rng(0).uniform(-.25, .25, (20000, 2))
check("(0,0) is a local minimum (F>=0 within 0.25)", min(F(p) for p in nb) >= -1e-12)


def subgd(x0, c=.05, steps=4000):
    x = np.array(x0, float)
    for k in range(steps):
        x = x - c / np.sqrt(k + 1) * F_sub(x)
    return x


x_bad, x_good = subgd([.15, .1]), subgd([.6, .85])
print(f"  trapped end {np.round(x_bad, 3)} F={F(x_bad):.3f}; good end {np.round(x_good, 3)} F={F(x_good):.3f}")
check("start near the bad min stays at F~0", abs(F(x_bad)) < .02)
check("start near beta=1 reaches F~-1", F(x_good) < -.95)
done()
