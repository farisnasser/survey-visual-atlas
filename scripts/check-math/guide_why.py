"""Guide chapter 11 (survey Section IV-D): the new 'normgd' visual, Arora, Li & Panigrahi (2022)'s Fig. 1 example
L(x, y) = (1 + x^2) y^2. Minimizers: the line y = 0. Hessian there: diag(0, 2(1 + x^2)), so sharpness 2(1 + x^2) is
smallest at x = 0. Normalized GD on L and GD on sqrt(L) oscillate around y = 0 and drift along it towards x = 0;
plain GD with a small step stops at the nearest minimizer and does not drift.
"""
import numpy as np
from common import check, check_grad, check_hess_at, done

print("guide/why")
L = lambda p: (1 + p[0] ** 2) * p[1] ** 2
gL = lambda p: np.array([2 * p[0] * p[1] ** 2, 2 * (1 + p[0] ** 2) * p[1]])
check_grad("L", L, gL, [[3, .4], [-1.2, .7], [.3, -.2]])
check_hess_at("L at (x, 0) = diag(0, 2(1 + x^2)), x = 3", L, np.array([3., 0.]), [[0, 0], [0, 20]])
sq = lambda p: np.sqrt(1 + p[0] ** 2) * abs(p[1])
gsq = lambda p: np.array([p[0] / np.sqrt(1 + p[0] ** 2) * abs(p[1]), np.sqrt(1 + p[0] ** 2) * np.sign(p[1])])
check_grad("sqrt(L) away from y = 0", sq, gsq, [[3, .4], [-1.2, .7], [.3, -.2]])

def run(kind, eta, T, p0=(3.0, 0.4)):
    p = np.array(p0, float); P = [p.copy()]
    for _ in range(T):
        if kind == 'ngd':
            g = gL(p); n = np.linalg.norm(g)
            if n == 0: break
            p = p - eta * g / n
        elif kind == 'sqrt':
            p = p - eta * gsq(p)
        else:
            p = p - eta * gL(p)
        P.append(p.copy())
    return np.array(P)

ETA, T = 0.1, 3000
A = run('ngd', ETA, T); B = run('sqrt', ETA, T); C = run('gd', 0.02, T)
sh = lambda x: 2 * (1 + x ** 2)
print(f"   normalized GD: x_T = {A[-1,0]:.4f}, sharpness {sh(A[-1,0]):.4f}, |y| last 100 <= {np.abs(A[-100:,1]).max():.3f}")
print(f"   GD on sqrt L : x_T = {B[-1,0]:.4f}, sharpness {sh(B[-1,0]):.4f}, |y| last 100 <= {np.abs(B[-100:,1]).max():.3f}")
print(f"   plain GD     : x_T = {C[-1,0]:.4f}, sharpness {sh(C[-1,0]):.4f}, |y| = {abs(C[-1,1]):.1e}")
check("start (3, 0.4): sharpness at the nearest minimizer is 20", sh(3.0) == 20)
check("normalized GD drifts to the flattest minimizer: x_3000 < 0.001", abs(A[-1, 0]) < 1e-3, f"{A[-1,0]:.2e}")
check("GD on sqrt(L) drifts to the flattest minimizer: x_1000 = 0.020, x_3000 < 1e-5", abs(B[1000, 0] - .0198) < 1e-3 and abs(B[-1, 0]) < 1e-5, f"{B[1000,0]:.4f}, {B[-1,0]:.1e}")
check("GD on sqrt(L): x_500 = 0.243", abs(B[500, 0] - .2425) < 1e-3, f"{B[500,0]:.4f}")
check("both keep oscillating across y = 0 (|y| stays of order eta)", 0.02 < np.abs(A[-100:, 1]).max() <= ETA + 1e-9 and 0.02 < np.abs(B[-100:, 1]).max() <= ETA)
sB = np.sign(B[-200:, 1])
check("GD on sqrt(L) ends in a period-2 orbit across the valley (y = 0.0267, -0.0733)", np.all(sB[1:] != sB[:-1]) and abs(B[-1, 1] + .0733) < 1e-3 and abs(B[-2, 1] - .0267) < 1e-3, f"{B[-2,1]:.4f}, {B[-1,1]:.4f}")
check("plain GD (eta = 0.02) stops near x = 2.97 with sharpness 19.6", abs(C[-1, 0] - 2.97) < .01 and abs(sh(C[-1, 0]) - 19.64) < .05, f"{C[-1,0]:.4f}, {sh(C[-1,0]):.3f}")
# milestones quoted in the text: normalized GD x after 500 and 1000 steps
check("normalized GD: x_500 = 2.13, x_1000 = 1.01", abs(A[500, 0] - 2.13) < .01 and abs(A[1000, 0] - 1.01) < .01, f"{A[500,0]:.3f}, {A[1000,0]:.3f}")
done()
