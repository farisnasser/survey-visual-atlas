"""Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky 2024: the road less scheduled (Schedule-Free SGD).

Eqs. (3)-(5): y_t = (1 - beta) z_t + beta x_t;  z_{t+1} = z_t - gamma grad f(y_t, xi_t);  x_{t+1} = (1 - c_{t+1}) x_t + c_{t+1} z_{t+1},
c_{t+1} = 1/(t + 1), z_1 = x_1. beta = 0: Polyak-Ruppert averaging; beta = 1: primal averaging.
Theorem 1: convex, G-Lipschitz, gamma = D/(G sqrt T): E[F(x_T) - F(x*)] <= DG/sqrt(T) for any beta in [0, 1].
Toy for the page: noisy anisotropic least squares in 2-D (3D tab) and a 20-D noisy least-squares regression (anytime comparison).
"""
from common import *

print("defazio2024road")
# notes' worked example: f = 1/2 (x - 2)^2, gamma = 0.5, beta = 0.8, z1 = x1 = 0
g = lambda x: x - 2
z, x = 0., 0.
y = .2 * z + .8 * x; z = z - .5 * g(y); x = .5 * x + .5 * z
check("notes step 1: y1 = 0, z2 = 1, x2 = 0.5", y == 0 and z == 1 and x == .5)
y = .2 * z + .8 * x; z = z - .5 * g(y); x = (2 / 3) * x + (1 / 3) * z
check("notes step 2: y2 = 0.6, z3 = 1.7, x3 = 0.9", abs(y - .6) < 1e-12 and abs(z - 1.7) < 1e-12 and abs(x - .9) < 1e-12)
check("c_{t+1} = 1/(t+1) makes x_T the plain average of z_1..z_T", True)
zs = np.random.default_rng(0).normal(size=50); xx = zs[0]
for t in range(1, 50): xx = (1 - 1 / (t + 1)) * xx + zs[t] / (t + 1)
check("x_T = mean(z_1..z_T) numerically", abs(xx - zs.mean()) < 1e-12)

# ---------- 20-D noisy least squares, anytime comparison ----------
rng = np.random.default_rng(1)
d, N = 20, 2000
A = rng.normal(size=(N, d)) * np.linspace(2, .2, d)
wst = rng.normal(size=d); b = A @ wst + .5 * rng.normal(size=N)
F = lambda w: .5 * np.mean((A @ w - b) ** 2)
wopt = np.linalg.lstsq(A, b, rcond=None)[0]; Fopt = F(wopt)
H = A.T @ A / N; Lmax = eig(H)[-1]


def sg(w, idx):
    a = A[idx]; return a.T @ (a @ w - b[idx]) / len(idx)


def run(kind, T, gamma, beta=.9, bs=4, seed=7):
    r = np.random.default_rng(seed); w = np.zeros(d); z = w.copy(); xa = w.copy(); out = []
    for t in range(1, T + 1):
        idx = r.integers(0, N, bs)
        if kind == "sf":
            yv = (1 - beta) * z + beta * xa; z = z - gamma * sg(yv, idx); xa = (1 - 1 / (t + 1)) * xa + z / (t + 1); cur = xa
        elif kind == "const":
            w = w - gamma * sg(w, idx); cur = w
        elif kind == "lin":
            w = w - gamma * (1 - (t - 1) / T) * sg(w, idx); cur = w
        out.append(F(cur) - Fopt)
    return np.array(out)


TS = [250, 1000, 4000]
gam_c = [.005, .01, .02, .04, .08, .16, .32]
best_lin = {}
for T in TS:
    curves = {gm: c for gm in gam_c if np.isfinite((c := run("lin", T, gm))[-1])}
    gm = min(curves, key=lambda k: curves[k][-1]); best_lin[T] = (gm, curves[gm])
gsf = {gm: c for gm in gam_c if np.all(np.isfinite(c := run("sf", max(TS), gm)))}
gm_sf = min(gsf, key=lambda k: np.mean([gsf[k][T - 1] for T in TS])); SF = gsf[gm_sf]
const = {gm: c for gm in gam_c if np.all(np.isfinite(c := run("const", max(TS), gm)))}; gm_c = min(const, key=lambda k: const[k][-1]); CONST = const[gm_c]
for T in TS:
    print(f"    T={T}: linear decay (tuned for T, gamma {best_lin[T][0]}) {best_lin[T][1][-1]:.3e} | schedule-free (one run, gamma {gm_sf}) at T {SF[T - 1]:.3e} | constant SGD (gamma {gm_c}) at T {CONST[T - 1]:.3e}")
ratio = [SF[T - 1] / best_lin[T][1][-1] for T in TS]
check("schedule-free, one run without knowing T, is within 2x of linear decay tuned for each T", all(r_ < 2 for r_ in ratio), f"ratios {np.round(ratio, 2)}")
check("schedule-free beats constant-step SGD at the longest horizon", SF[-1] < CONST[-1], f"{SF[-1]:.2e} vs {CONST[-1]:.2e}")
save_json("defazio2024road", "data.json", {"Ts": TS, "sf": {"gamma": gm_sf, "gap": sig(SF[::10], 4)}, "const": {"gamma": gm_c, "gap": sig(CONST[::10], 4)},
                                           "lin": {str(T): {"gamma": best_lin[T][0], "gap": sig(best_lin[T][1][::10], 4), "end": float(f"{best_lin[T][1][-1]:.4g}"), "sf": float(f"{SF[T - 1]:.4g}")} for T in TS}})
# ---------- 3D tab toy ----------
A2 = np.diag([2.0, .25])
f2 = lambda w: .5 * w @ A2 @ w
check_grad("3D toy 1/2 w^T diag(2, 0.25) w", f2, lambda w: A2 @ w, rand_pts(4, -2, 2))
done()
