"""Guide chapter 12 (survey Section IV-E): the new 'centralflow' visual.

Toy loss in three coordinates (an extension of Damian et al. 2023's App. B toy, eta = 1):
    L(x, y, z) = (2/eta + y) x^2 / 2 + V(y, z),   V = k (y - m z)^2 / 2 + e (z - z0)^2 / 2.
x is the direction GD oscillates along; its curvature is S = 2/eta + y, so y = 0 is the edge S = 2/eta.
V pulls (y, z) towards (m z0, z0), a point with y > 0: gradient flow walks into the region that is too sharp.
Central flow (Cohen et al. 2025, Eq. 14-15 with one unstable direction): x = 0 on average, and once y reaches 0 with
the flow pushing y up, the y-component of the gradient is projected out (grad S = e_y), so it slides along y = 0.
Predicted oscillation variance: sigma^2 = 2 <grad S, -grad L> / |grad S|^2 = -2 dV/dy at y = 0.
"""
import numpy as np
from common import check, check_grad, done

print("guide/flow")
ETA, K, E, M, Z0 = 1.0, 0.05, 0.005, 0.5, 1.0
X0, Y0, ZS, T = 1e-2, -0.3, -0.5, 3000

def V(y, z): return K * (y - M * z) ** 2 / 2 + E * (z - Z0) ** 2 / 2
def dV(y, z): return np.array([K * (y - M * z), E * (z - Z0) - K * M * (y - M * z)])
def Lf(p): x, y, z = p; return (2 / ETA + y) * x * x / 2 + V(y, z)
def gLf(p): x, y, z = p; d = dV(y, z); return np.array([(2 / ETA + y) * x, x * x / 2 + d[0], d[1]])
check_grad("toy L(x, y, z)", Lf, gLf, [[.1, -.2, .3], [-.05, .1, -.4], [.2, .3, .9]])

# gradient descent (exactly what the visual runs)
p = np.array([X0, Y0, ZS]); GD = [p.copy()]
for _ in range(T): p = p - ETA * gLf(p); GD.append(p.copy())
GD = np.array(GD)

# gradient flow on V with x = 0 (dt = 1 step, RK4 with 10 substeps), and central flow
def rk4(f, q, h):
    a = f(q); b = f(q + h / 2 * a); c = f(q + h / 2 * b); d = f(q + h * c); return q + h / 6 * (a + 2 * b + 2 * c + d)
gf = lambda q: -ETA * dV(*q)
def cf(q):
    v = -ETA * dV(*q)
    if q[0] >= -1e-12 and v[0] > 0: v = np.array([0.0, v[1]])   # on the edge and pushed up: project out e_y
    return v
GF = [np.array([Y0, ZS])]; CF = [np.array([Y0, ZS])]
q1, q2 = GF[0].copy(), CF[0].copy()
for _ in range(T):
    for _ in range(10):
        q1 = rk4(gf, q1, .1)
        q2 = rk4(cf, q2, .1)
        if q2[0] > 0: q2[0] = 0.0
    GF.append(q1.copy()); CF.append(q2.copy())
GF, CF = np.array(GF), np.array(CF)

z_edge = E * Z0 / (E + K * M * M)           # minimizer of V on y = 0
sig2 = -2 * dV(0.0, z_edge)[0]               # = 2 K M z_edge
t_hit = int(np.argmax(CF[:, 0] >= -1e-9))
print(f"   GD end: y = {GD[-1,1]:+.2e}, z = {GD[-1,2]:.4f}, |x| = {abs(GD[-1,0]):.4f}, mean x^2 (last 400) = {np.mean(GD[-400:,0]**2):.5f}")
print(f"   CF end: y = {CF[-1,0]:+.2e}, z = {CF[-1,1]:.4f}; reaches the edge at step {t_hit}")
print(f"   GF end: y = {GF[-1,0]:.4f}, z = {GF[-1,1]:.4f}; sharpness 2/eta + y = {2+GF[-1,0]:.4f}")
check("gradient flow ends in the sharp region: y -> m z0 = 0.5, S -> 2.5", abs(GF[-1, 0] - .5) < 5e-3 and abs(GF[-1, 1] - 1) < 1e-2, f"{GF[-1]}")
check("edge minimizer z = e z0/(e + k m^2) = 0.2857", abs(z_edge - 0.285714) < 1e-5)
check("central flow ends at (y, z) = (0, 0.2857)", abs(CF[-1, 0]) < 1e-9 and abs(CF[-1, 1] - z_edge) < 2e-3, f"{CF[-1]}")
check("GD ends at the same point: y ~ 0, z = 0.2857", abs(GD[-1, 1]) < 1e-3 and abs(GD[-1, 2] - z_edge) < 2e-3)
check("predicted variance sigma^2 = 2 k m z = 0.01429; GD's time-averaged x^2 matches", abs(sig2 - .014286) < 1e-5 and abs(np.mean(GD[-400:, 0] ** 2) - sig2) < 3e-4, f"{np.mean(GD[-400:,0]**2):.5f}")
check("GD sharpness 2/eta + y_t stays within 0.01 of 2 over the last 1000 steps", np.max(np.abs(GD[-1000:, 1])) < .01, f"{np.max(np.abs(GD[-1000:,1])):.2e}")
# Before the edge, GD and the central flow (= gradient flow there) agree; both reach y = 0 at step 123.
t_gd = int(np.argmax(GD[:, 1] > 0))
check("GD and the central flow reach the edge together (step 123)", t_gd == 123 and t_hit == 123, f"GD {t_gd}, CF {t_hit}")
d0 = np.max(np.abs(GD[:124, 2] - CF[:124, 1]))
check("before the edge, |z_GD - z_CF| < 0.002", d0 < .002, f"{d0:.4f}")
# While y < 0 the oscillation along x decays (multiplier |1 + y| < 1): |x| = 3.2e-11 at the crossing.
check("|x| has decayed to ~3e-11 when the edge is reached", 1e-11 < abs(GD[123, 0]) < 1e-10, f"{abs(GD[123,0]):.2e}")
# So GD overshoots into the sharp region while the oscillation regrows, then self-stabilizes back onto the flow
k = int(np.argmax(GD[:, 1]))
check("overshoot: GD's sharpness peaks at 2/eta + 0.242 at step 293", abs(GD[k, 1] - .242) < 1e-3 and k == 293, f"{GD[k,1]:.4f} at {k}")
d1 = np.max(np.abs(GD[600:, 2] - CF[600:, 1]))
check("from step 600 on, |z_GD - z_CF| < 0.002 (GD back on the central flow)", d1 < .002, f"{d1:.4f}")
dzg = np.max(np.abs(GD[:, 2] - GF[:, 1]))
check("gradient flow departs from GD by more than 0.5 in z", dzg > .5, f"{dzg:.3f}")
# loss: central-flow loss decreases monotonically (Prop. 1), GD's loss does not
Lcf = np.array([V(*q) for q in CF]); Lgd = np.array([Lf(q) for q in GD]); Lgf = np.array([V(*q) for q in GF])
check("central-flow loss is non-increasing (Prop. 1)", np.all(np.diff(Lcf) <= 1e-12))
check("GD loss is non-monotone (goes up at some steps)", np.sum(np.diff(Lgd) > 0) > 10, f"{np.sum(np.diff(Lgd) > 0)} increases")
check("final losses: gradient flow 0, central flow V = 0.001786",
      Lgf[-1] < 1e-5 and abs(Lcf[-1] - 0.0017857) < 2e-6, f"GF {Lgf[-1]:.2e}, CF {Lcf[-1]:.6f}")
# GD's loss also contains the oscillation energy (2/eta + y) x^2 / 2 = x^2 at y = 0: 0.001786 + 0.01429 = 0.01607
check("GD's final loss = central-flow loss + oscillation energy x^2 = 0.01607", abs(Lgd[-1] - (Lcf[-1] + GD[-1, 0] ** 2)) < 1e-6 and abs(Lgd[-1] - .016071) < 1e-5, f"{Lgd[-1]:.6f}")
check("|x| settles at 0.1195 = sqrt(sigma^2)", abs(abs(GD[-1, 0]) - np.sqrt(sig2)) < 1e-4, f"{abs(GD[-1,0]):.4f}")
print(f"   central flow reaches the edge at step {t_hit}")
done()
