"""Crash course stop 9 (course/adaptive): preconditioning (diagonal vs Newton), AdaGrad / Adam on aligned and rotated
valleys, Adam's bias correction, the Adam / AMSGrad / ADOPT counterexample of Taniguchi et al. (with the page's
mulberry32 RNG, ported exactly), gradient clipping under (L0, L1)-smoothness, Shampoo's memory."""
import math

from common import *


class Mulberry:   # mulberry32 of src/lib/math.js (same port as course_noise.py)
    def __init__(self, seed=1):
        self.a = seed & 0xFFFFFFFF

    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.a
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296


def imul(a, b):
    return (a * b) & 0xFFFFFFFF


print("course_adaptive")
LC = 10.


def Hrot(kap, deg):
    th = np.radians(deg); c, s = np.cos(th), np.sin(th); R = np.array([[c, -s], [s, c]])
    return R @ np.diag([LC / kap, LC]) @ R.T


# ---- section 2: preconditioning. Newton (P = H^-1) vs the diagonal (Jacobi) preconditioner P = diag(H)^-1
def k_eff(H, P):
    S = np.real(np.linalg.eigvals(P @ H)); return S.max() / S.min()


for kap in (10, 100):
    ke = {deg: k_eff(Hrot(kap, deg), np.diag(1 / np.diag(Hrot(kap, deg)))) for deg in (0, 30, 45, 90)}
    print(f"  kappa = {kap}: effective condition number with the diagonal preconditioner: " + ", ".join(f"{d} deg {v:.2f}" for d, v in ke.items()))
check("diagonal preconditioner: kappa_eff = 1 for an axis-aligned valley (0 and 90 deg)", all(np.isclose(k_eff(Hrot(k, d), np.diag(1 / np.diag(Hrot(k, d)))), 1) for k in (10, 100) for d in (0, 90)))
check("... = kappa itself at 45 deg (both diagonal entries equal (mu + L)/2): no help at all", all(np.isclose(k_eff(Hrot(k, 45), np.diag(1 / np.diag(Hrot(k, 45)))), k) for k in (10, 100)))
check("kappa = 100 at 30 deg: kappa_eff = 75.5", round(k_eff(Hrot(100, 30), np.diag(1 / np.diag(Hrot(100, 30)))), 1) == 75.5)
check("Newton: P = H^-1 gives kappa_eff = 1 at any angle, and one step with eta = 1 lands on the minimum",
      all(np.isclose(k_eff(Hrot(100, d), np.linalg.inv(Hrot(100, d))), 1) for d in range(0, 91, 15)) and np.allclose(np.array([.3, -1.]) - np.linalg.inv(Hrot(100, 30)) @ Hrot(100, 30) @ np.array([.3, -1.]), 0))
z = np.array([.7, -.4]); P = np.diag([4., .25]); Hq = Hrot(100, 20)
y = np.linalg.inv(np.sqrt(P)) @ z
check("change of variables x = P^(1/2) y: GD on y is preconditioned GD on x with Hessian P^(1/2) H P^(1/2)",
      np.allclose(np.sqrt(P) @ (y - .1 * np.sqrt(P) @ Hq @ np.sqrt(P) @ y), z - .1 * P @ Hq @ z))
START_P = np.array([-1.3, 1.1])


def steps_pre(H, P, tol=1e-8, T=100000):
    S = np.real(np.linalg.eigvals(P @ H)); eta = 2 / (S.min() + S.max()); x = START_P.copy()
    for t in range(T):
        if .5 * x @ H @ x < tol: return t
        x = x - eta * P @ (H @ x)
    return T


sp = {d: (steps_pre(Hrot(100, d), np.eye(2)), steps_pre(Hrot(100, d), np.diag(1 / np.diag(Hrot(100, d))))) for d in (0, 30, 45)}
print("  precond visual (kappa = 100, best step each): steps to f < 1e-8 (GD, diagonal): " + str(sp))
check("precond visual kappa = 100: GD 506 / 525 / 528 steps at 0 / 30 / 45 deg; diagonal 1 / 396 / 528",
      sp[0] == (506, 1) and sp[30] == (525, 396) and sp[45] == (528, 528), str(sp))

# ---- sections 3-5: AdaGrad and Adam on the kappa = 100 valley, aligned vs rotated 45 deg, 400 steps
START_R = np.array([-1.3, 1.2])


def race(kind, H, T=400, lr=.05, b1=.9, b2=.999, eps=1e-8):
    x = START_R.copy(); m = np.zeros(2); v = np.zeros(2); G = np.zeros(2); mu, L = np.linalg.eigvalsh(H)
    for t in range(1, T + 1):
        g = H @ x
        if kind == "gd": x = x - 2 / (L + mu) * g
        elif kind == "adagrad": G = G + g * g; x = x - lr * g / (np.sqrt(G) + eps)
        else:
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return .5 * x @ H @ x


rr = {d: (race("gd", Hrot(100, d)), race("adagrad", Hrot(100, d), lr=.5), race("adam", Hrot(100, d), lr=.05)) for d in (0, 45)}
print("  race, f after 400 steps (GD best step, AdaGrad 0.5, Adam 0.05): " + str({d: [f"{v:.1e}" for v in r_] for d, r_ in rr.items()}))
check("aligned valley: AdaGrad and Adam end far below GD (GD 8.2e-7, AdaGrad 5.6e-123, Adam 1.2e-18)",
      f"{rr[0][0]:.1e}" == "8.2e-07" and f"{rr[0][1]:.1e}" == "5.6e-123" and f"{rr[0][2]:.1e}" == "1.2e-18")
check("rotated 45 deg: the advantage is gone, both end above GD (GD 1.8e-6, AdaGrad 2.1e-5, Adam 5.6e-5)",
      f"{rr[45][0]:.1e}" == "1.8e-06" and f"{rr[45][1]:.1e}" == "2.1e-05" and f"{rr[45][2]:.1e}" == "5.6e-05")

# ---- section 5: Adam's bias correction (constant gradient)
check("EMA started at 0 with constant input g: m_t = (1 - b1^t) g", np.isclose(sum((1 - .9) * .9 ** (5 - s) for s in range(1, 6)), 1 - .9 ** 5))
for b2, pk, tp in ((.99, 2.131, 13), (.999, 6.569, 12), (.9999, 20.72, 12)):
    t = np.arange(1, 100001); s = (1 - .9 ** t) / np.sqrt(1 - b2 ** t)
    check(f"uncorrected step / alpha = (1 - 0.9^t)/sqrt(1 - b2^t): beta2 = {b2}: peak {pk} at t = {tp}, first step {s[0]:.3g}",
          round(s.max(), 3 if b2 < .9999 else 2) == pk and t[s.argmax()] == tp)
check("first uncorrected step: 0.1/sqrt(0.001) = 3.16 alpha; corrected step = alpha exactly", round(.1 / np.sqrt(.001), 2) == 3.16 and np.isclose((1 - .9) / (1 - .9) / np.sqrt((1 - .999) / (1 - .999)), 1))

# ---- section 6: the counterexample of Taniguchi et al. 2024 (Eq. 17), one seeded gradient stream shared by the three methods
def stream(k, T, seed=1):
    r = Mulberry(seed); return np.array([k * k if r() < 1 / k else -k for _ in range(T + 2)], float)


check("toy gradient is unbiased with mean 1 (so theta* = -1 on [-1, 1]): k^2 (1/k) - k (1 - 1/k) = 1", all(np.isclose(k * k / k - k * (1 - 1 / k), 1) for k in (10, 50)))


def toy(k, b2s, T=400000, b1=.9, eps=1e-8):
    G = stream(k, T); b2 = np.array(b2s); n = len(b2)
    tha = np.zeros(n); ma = np.zeros(n); va = np.zeros(n); ths = np.zeros(n); ms = np.zeros(n); vs = np.zeros(n); vmax = np.zeros(n)
    tho = np.zeros(n); vo = np.full(n, G[0] ** 2); mo = G[1] / np.maximum(np.sqrt(vo), eps)
    for t in range(1, T + 1):
        eta = .01 / math.sqrt(1 + .01 * t); g = G[t]
        ma = b1 * ma + (1 - b1) * g; va = b2 * va + (1 - b2) * g * g
        tha = np.clip(tha - eta * (ma / (1 - b1 ** t)) / (np.sqrt(va / (1 - b2 ** t)) + eps), -1, 1)
        ms = b1 * ms + (1 - b1) * g; vs = b2 * vs + (1 - b2) * g * g; vmax = np.maximum(vmax, vs / (1 - b2 ** t))
        ths = np.clip(ths - eta * (ms / (1 - b1 ** t)) / (np.sqrt(vmax) + eps), -1, 1)
        tho = np.clip(tho - eta * mo, -1, 1); vo = b2 * vo + (1 - b2) * g * g; mo = b1 * mo + (1 - b1) * G[t + 1] / np.maximum(np.sqrt(vo), eps)
    return tha, ths, tho


B2S = [.1, .5, .9, .99, .999]
a10, s10, o10 = toy(10, B2S)
print("  k = 10, final theta for beta2 = 0.1/0.5/0.9/0.99/0.999: Adam", a10.round(3), "AMSGrad", s10.round(3), "ADOPT", o10.round(3))
check("k = 10: Adam ends at the worst point +1 for beta2 <= 0.9 and near -1 for 0.99, 0.999 (-0.992, -0.994)",
      np.all(a10[:3] > .999) and list(a10[3:].round(3)) == [-.992, -.994])
check("k = 10: AMSGrad -0.998, ADOPT -0.980 / -0.981 / -0.991 / -0.994 / -0.994", np.allclose(s10.round(3), -.998) and list(o10.round(3)) == [-.98, -.981, -.991, -.994, -.994])
a50, s50, o50 = toy(50, B2S)
print("  k = 50: Adam", a50.round(3), "AMSGrad", s50.round(3), "ADOPT", o50.round(3))
check("k = 50: Adam fails for every beta2 (ends >= 0.19); AMSGrad -0.052; ADOPT <= -0.966", np.all(a50 > .19) and np.allclose(s50.round(3), -.052) and np.all(o50 < -.965))

# ---- section 7: clipping on f = x^2/2 + x^4/4 + y^2/2 (Zhang et al. 2020's kind of example), eps = 1e-4
FZ = lambda z: z[0] ** 2 / 2 + z[0] ** 4 / 4 + z[1] ** 2 / 2; GZ = lambda z: np.array([z[0] + z[0] ** 3, z[1]])
check_grad("x^2/2 + x^4/4 + y^2/2", FZ, GZ, rand_pts(5, -3, 3))
PZ = rand_pts(5000, -6, 6, seed=2)
check("(L0, L1) = (1, 3)-smooth on [-6, 6]^2: curvature 1 + 3x^2 <= 1 + 3 ||grad||", all(max(1 + 3 * p[0] ** 2, 1) <= 1 + 3 * np.linalg.norm(GZ(p)) for p in PZ))
check("not L-smooth: curvature 1 + 3x^2 is 193 at x = 8 and unbounded", 1 + 3 * 64 == 193)


def iters(step, z0, cap=200000):
    zz = np.array(z0, float)
    for t in range(cap):
        g = GZ(zz)
        if np.linalg.norm(g) <= 1e-4: return t
        zz = step(zz, g)
        if not np.all(np.isfinite(zz)) or np.abs(zz).max() > 1e6: return None
    return None


HS = np.geomspace(1e-4, 1.9, 60); table = {}
for x0 in (1, 2, 4, 8):
    best = min((n, h) for h in HS if (n := iters(lambda zz, g, h=h: zz - h * g, [x0, 1.])) is not None)
    nc = iters(lambda zz, g: zz - min(1., .5 / np.linalg.norm(g)) * g, [x0, 1.])
    table[x0] = (round(float(np.linalg.norm(GZ([x0, 1.]))), 1), best[0], round(float(best[1]), 4), nc)
print("  clipping table x0: (M, best fixed-step iterations, best step, clipped iterations):", table)
check("best fixed step needs 6 / 22 / 81 / 322 iterations from x0 = 1 / 2 / 4 / 8; clipped GD (eta 1, gamma 0.5) 4 / 6 / 10 / 18",
      [table[x][1] for x in (1, 2, 4, 8)] == [6, 22, 81, 322] and [table[x][3] for x in (1, 2, 4, 8)] == [4, 6, 10, 18])
check("viz constants: best steps 0.9742 / 0.3577 / 0.1111 / 0.0292, M = 2.2 / 10.0 / 68.0 / 520.0",
      [table[x][2] for x in (1, 2, 4, 8)] == [.9742, .3577, .1111, .0292] and [table[x][0] for x in (1, 2, 4, 8)] == [2.2, 10., 68., 520.])
check("stability of a fixed step near x0 = 8 needs h < 2/193 = 0.0104", round(2 / 193, 4) == .0104)

# ---- section 8: Shampoo's memory for an m x n weight matrix (m = n = 1000)
m_, n_ = 1000, 1000
check("memory: diagonal m n = 10^6, Shampoo m^2 + n^2 = 2 x 10^6, full-matrix AdaGrad (m n)^2 = 10^12", m_ * n_ == 10 ** 6 and m_ ** 2 + n_ ** 2 == 2 * 10 ** 6 and (m_ * n_) ** 2 == 10 ** 12)
A = np.random.default_rng(0).normal(size=(3, 3)); A = A @ A.T + np.eye(3); B = np.random.default_rng(1).normal(size=(4, 4)); B = B @ B.T + np.eye(4)
Gm = np.random.default_rng(2).normal(size=(3, 4))
check("Kronecker identity behind Shampoo: (L kron R^T) vec(G) = vec(L G R)", np.allclose(np.kron(A, B.T) @ Gm.reshape(-1), (A @ Gm @ B).reshape(-1)))
done()
