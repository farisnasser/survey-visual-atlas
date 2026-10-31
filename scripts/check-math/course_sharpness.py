"""Crash course stop 10 (course/sharpness): sharpness, stability 2/eta, edge of stability, self-stabilization,
the reparameterization caveat, SAM and warmup. Every number on the page is recomputed here."""
import json
from common import *

print("course_sharpness")

# ---------- 1. two equally deep wells, one sharp and one flat (visual 'wells') ----------
S_C, F_C, SS, SF, AS, AF, Q = np.array([-.9, .35]), np.array([.8, -.2]), .22, .75, 1.0, 1.045, .06


def wL(p):
    ds, df = p - S_C, p - F_C
    return Q * p @ p - AS * np.exp(-ds @ ds / (2 * SS ** 2)) - AF * np.exp(-df @ df / (2 * SF ** 2))


def wG(P):   # vectorized over rows
    P = np.atleast_2d(P); ds, df = P - S_C, P - F_C
    es = AS * np.exp(-(ds ** 2).sum(1) / (2 * SS ** 2)); ef = AF * np.exp(-(df ** 2).sum(1) / (2 * SF ** 2))
    return 2 * Q * P + es[:, None] * ds / SS ** 2 + ef[:, None] * df / SF ** 2


def wH(p):
    ds, df = p - S_C, p - F_C; I = np.eye(2)
    es = AS * np.exp(-ds @ ds / (2 * SS ** 2)); ef = AF * np.exp(-df @ df / (2 * SF ** 2))
    return 2 * Q * I + es * (I / SS ** 2 - np.outer(ds, ds) / SS ** 4) + ef * (I / SF ** 2 - np.outer(df, df) / SF ** 4)


check_grad("two wells", wL, lambda p: wG(p)[0], rand_pts(20, -2, 2))
for p in rand_pts(3, -1.5, 1.5, seed=3):
    check_hess_at(f"two wells at {np.round(p, 2)}", wL, p, wH(p))


def newton(p, n=60):
    for _ in range(n): p = p - np.linalg.solve(wH(p), wG(p)[0])
    return p


MS, MF = newton(S_C.copy()), newton(F_C.copy())
SHARP, FLAT = eig(wH(MS)).max(), eig(wH(MF)).max()
print(f"  sharp min {MS.round(4)} f = {wL(MS):.4f} S = {SHARP:.3f}; flat min {MF.round(4)} f = {wL(MF):.4f} S = {FLAT:.3f}")
check("wells: depths -1.008 and -1.007 (equal to 0.001)", round(wL(MS), 3) == -1.008 and round(wL(MF), 3) == -1.007)
check("wells: sharpness 20.8 vs 1.97 (ratio about 10.6)", round(SHARP, 1) == 20.8 and round(FLAT, 2) == 1.97 and round(SHARP / FLAT, 1) == 10.6)
check("wells: thresholds 2/S = 0.096 and 1.01", round(2 / SHARP, 3) == .096 and round(2 / FLAT, 2) == 1.01)
check("wells: both are minima (Hessian positive definite)", eig(wH(MS)).min() > 0 and eig(wH(MF)).min() > 0)
GX, GY = np.meshgrid(np.linspace(-2, 2, 21), np.linspace(-1.5, 1.5, 17)); P0 = np.c_[GX.ravel(), GY.ravel()]


def fates(eta, T=4000):
    P = P0.copy()
    with np.errstate(all="ignore"):
        for _ in range(T): P = P - eta * wG(P)
    a = int((np.linalg.norm(P - MS, axis=1) < 1e-3).sum()); b = int((np.linalg.norm(P - MF, axis=1) < 1e-3).sum())
    return a, b, len(P) - a - b


FATE = {e: fates(e) for e in (.05, .09, .1, .15, .2, .25, .3, .6, 1.0, 1.05)}
print("  fates (sharp, flat, never settled):", FATE)
check("wells: 357 starts; for eta < 0.096: 87 settle in the sharp well, 270 in the flat one", FATE[.05] == (87, 270, 0) and FATE[.09] == (87, 270, 0))
check("wells: eta = 0.1 to 0.2: the 87 keep bouncing in the sharp well, none settle there", all(FATE[e] == (0, 270, 87) for e in (.1, .15, .2)))
check("wells: eta = 0.25 to 1.0: all 357 end in the flat well", all(FATE[e] == (0, 357, 0) for e in (.25, .3, .6, 1.0)))
check("wells: eta = 1.05 > 2/1.97: nothing settles", FATE[1.05] == (0, 0, 357))
ok = True
for e in np.round(np.arange(.02, 1.0101, .005), 3):
    P = gd(lambda p: wG(p)[0], [-1.25, .75], e, 4000)
    ok &= bool(np.abs(P).max() < 50)
check("wells: the ball's GD path from (-1.25, 0.75) stays inside the ball's box [-50, 50]^2 for every eta <= 1.01 (no clamping)", ok)
check("wells: default ball start (-1.25, 0.75) ends sharp at eta 0.05, flat at 0.3",
      np.linalg.norm(gd(lambda p: wG(p)[0], [-1.25, .75], .05, 4000)[-1] - MS) < 1e-3 and np.linalg.norm(gd(lambda p: wG(p)[0], [-1.25, .75], .3, 4000)[-1] - MF) < 1e-3)

# ---------- 2. stability at a minimum: the 1-D rule (text) ----------
check("text: curvature 20 at eta 0.11: factor 1 - 2.2 = -1.2 (grows 20%/step); curvature 2: factor 0.78", np.isclose(1 - .11 * 20, -1.2) and np.isclose(1 - .11 * 2, .78))

# ---------- 3. a valley of equivalent minima: f = 1/2 (uv - 1)^2 (visuals 'family' and 'sam') ----------
uvL = lambda p: .5 * (p[0] * p[1] - 1) ** 2
uvG = lambda p: np.array([(p[0] * p[1] - 1) * p[1], (p[0] * p[1] - 1) * p[0]])
uvH = lambda p: np.array([[p[1] ** 2, 2 * p[0] * p[1] - 1], [2 * p[0] * p[1] - 1, p[0] ** 2]])
check_grad("valley 1/2 (uv - 1)^2", uvL, uvG, rand_pts(20, -2, 3))
check_hess_at("valley at (1.3, -0.4)", uvL, [1.3, -.4], uvH([1.3, -.4]))
for a in (.5, 1, 2, 4):
    ev = eig(uvH([a, 1 / a]))
    assert np.allclose(ev, [0, a * a + 1 / a ** 2]), ev
check("sam visual labels: sharpness of (a, 1/a) for a = 1, 1.5, 2, 2.5 is 2.00, 2.69, 4.25, 6.41", [round(a * a + 1 / a ** 2, 2) for a in (1, 1.5, 2, 2.5)] == [2.0, 2.69, 4.25, 6.41])
check("valley: at (a, 1/a) the Hessian eigenvalues are 0 and a^2 + 1/a^2: 2 at (1, 1), 4.25 at (2, 0.5), 16.06 at (4, 0.25)",
      np.isclose(1 + 1, 2) and np.isclose(4 + .25, 4.25) and round(16 + 1 / 16, 2) == 16.06)


def run_uv(p, eta, T=4000, mode="gd", rho=0.):
    p = np.array(p, float); S = []
    for _ in range(T):
        g = uvG(p)
        if mode == "gd": q = p
        elif mode == "un": q = p + rho * g
        else:
            n = np.linalg.norm(g); q = p + (rho * g / n if n > 1e-12 else 0)
        p = p - eta * uvG(q)
        if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e6: return None, None
        S.append(p @ p)
    return p, np.array(S)


START = [2.5, .05]
check("family: Hessian at the start (2.5, 0.05) has lambda_max 6.34", round(eig(uvH(START)).max(), 2) == 6.34)
END = {}
for eta in (.05, .1, .2, .3, .35, .4, .44, .45, .5):
    p, S = run_uv(START, eta)
    END[eta] = None if p is None else (round(p @ p, 2), abs(p[0] * p[1] - 1))
print("  family end sharpness:", {k: (v and v[0]) for k, v in END.items()})
check("family: eta 0.05 / 0.3 / 0.4 / 0.5 end at sharpness 6.54 / 5.42 / 2.54 / 2.01",
      [END[e][0] for e in (.05, .3, .4, .5)] == [6.54, 5.42, 2.54, 2.01] and all(END[e][1] < 1e-9 for e in (.05, .3, .4, .5)))
check("family: every converged run ends with sharpness below 2/eta", all(v[0] < 2 / e for e, v in END.items() if v))
def uv_path(p, eta, T=4000):
    p = np.array(p, float); out = [p.copy()]
    for _ in range(T): p = p - eta * uvG(p); out.append(p.copy())
    return np.array(out)


grid = np.round(np.arange(.02, .4601, .005), 3)
inside = all((lambda P: P[:, 0].min() >= -1 and P[:, 0].max() <= 3 and P[:, 1].min() >= -1 and P[:, 1].max() <= 2.2)(uv_path(START, e)) for e in grid)
check("family: for every slider value eta in [0.02, 0.46] the GD path stays inside the drawn square [-1, 3] x [-1, 2.2] (no clamping)", inside)
check("family: eta 0.44 ends at sharpness 2.00 (the flattest point (1, 1))", round(uv_path(START, .44)[-1] @ uv_path(START, .44)[-1], 2) == 2.0)

# ---------- 4. edge of stability: the Cohen-page MLP run (papers/cohen2021gradient/data.json) ----------
D = json.load(open(os.path.join(ROOT, "papers", "cohen2021gradient", "data.json")))
R = {float(k): np.array(v) for k, v in D["curves"].items()}
check("EoS data: tanh MLP 10-48-48-1, 128 points, MSE, full-batch GD", D["setup"] == "tanh MLP 10-48-48-1, 128 points, MSE, full-batch GD")
check("EoS: initial sharpness 4.05, initial loss 0.444", all(round(R[e][0, 2], 2) == 4.05 and round(R[e][0, 1], 3) == .444 for e in R))
check("EoS: eta = 0.1 (2/eta = 20): sharpness rises to 7.8 and never reaches 20; loss never increases",
      round(R[.1][:, 2].max(), 1) == 7.8 and int((np.diff(R[.1][:, 1]) > 0).sum()) == 0)
for e, s in ((.3, 6.66), (.36, 5.55), (.42, 4.75)):
    late = R[e][-100:, 2].mean()
    check(f"EoS: eta = {e}: last 1000 steps average sharpness {s} vs 2/eta = {2 / e:.2f}", round(late, 2) == s and abs(late - 2 / e) < .02)
check("EoS: peak sharpness above 2/eta before settling: 7.33 / 6.84 / 5.84",
      [round(R[e][:, 2].max(), 2) for e in (.3, .36, .42)] == [7.33, 6.84, 5.84])
check("EoS: loss rises in 73 / 42 / 96 of the 300 ten-step windows, yet ends below 1e-5",
      [int((np.diff(R[e][:, 1]) > 0).sum()) for e in (.3, .36, .42)] == [73, 42, 96] and all(R[e][-1, 1] < 1e-5 for e in (.3, .36, .42)))

# ---------- 5. self-stabilization: Damian et al. App. B toy restricted to (x, y), eta = 1, beta = 1 ----------
ETA, AL = 1.0, .002
dL = lambda z, al=AL: (2 / ETA + z[1]) * z[0] ** 2 / 2 - al * z[1]
dG = lambda z, al=AL: np.array([(2 / ETA + z[1]) * z[0], z[0] ** 2 / 2 - al])
check_grad("self-stabilization toy", dL, dG, rand_pts(10, -.4, .4))
check("toy: curvature across the trough d2L/dx2 = 2/eta + y", abs(fd_hess(dL, [.1, .13])[0, 0] - (2 + .13)) < 1e-5)


def dam(al, T=600, z0=(.02, -.05)):
    z = np.array(z0, float); out = [z.copy()]
    for _ in range(T): z = z - ETA * dG(z, al); out.append(z.copy())
    return np.array(out)


Z = dam(AL)
check("toy: GD is exactly Eq. (3): x' = -(1 + eta y) x, y' = y + eta (alpha - x^2/2)",
      np.allclose(Z[1:, 0], -(1 + ETA * Z[:-1, 1]) * Z[:-1, 0], atol=1e-15) and np.allclose(Z[1:, 1], Z[:-1, 1] + ETA * (AL - Z[:-1, 0] ** 2 / 2), atol=1e-15))
up = int(np.argmax(Z[:, 1] > 0))
print(f"  toy: S crosses 2/eta at step {up}; S range {2 + Z[:, 1].min():.3f}..{2 + Z[:, 1].max():.3f}; max |x| {np.abs(Z[:, 0]).max():.3f}")
check("toy (alpha 0.002, start x 0.02, S 1.95): S crosses 2/eta at step 27, stays in [1.80, 2.16], |x| <= 0.29",
      up == 27 and round(2 + Z[:, 1].min(), 2) == 1.80 and round(2 + Z[:, 1].max(), 2) == 2.16 and np.abs(Z[:, 0]).max() < .29)
dlt = np.sqrt(2 * AL)
check("toy: delta = sqrt(2 alpha / beta) = 0.063 (the |x| where the sharpness stops rising)", round(dlt, 3) == .063)
check("toy: average S over steps 100-600 is 1.99 (within 0.01 of 2/eta)", round(2 + Z[100:, 1].mean(), 2) == 1.99, f"{2 + Z[100:, 1].mean():.4f}")
for al in (.001, .0015, .0025, .003):
    Za = dam(al); assert np.abs(Za[:, 0]).max() < 1 and abs(Za[100:, 1].mean()) < .03, (al, np.abs(Za[:, 0]).max(), Za[100:, 1].mean())
ok = True
for al in np.round(np.arange(.001, .00301, .0001), 4):
    Za = dam(al); ok &= bool(np.abs(Za[:, 0]).max() < .45 and Za[:, 1].min() > -.35 and Za[:, 1].max() < .3)
check("toy: bounded for alpha in [0.001, 0.003] (slider range), mean S within 0.03 of 2/eta; path inside the drawn box (|x| < 0.45, -0.35 < y < 0.3)", ok)

# ---------- 6. Dinh et al.: same function, any sharpness ----------
check("text: (a, 1/a) all give uv = 1; a = 10 gives sharpness 100.01", round(100 + 1 / 100, 2) == 100.01)

# ---------- 7. SAM ----------
p, S = run_uv([2.2, .1], .3)
check("SAM visual: GD from (2.2, 0.1), eta 0.3 ends at sharpness 4.91", round(S[-1], 2) == 4.91)
for rho, s in ((.05, 4.20), (.1, 2.96), (.2, 2.00)):
    q, Sq = run_uv([2.2, .1], .3, mode="un", rho=rho)
    check(f"unnormalized SAM rho = {rho}: ends at sharpness {s} (Agarwala page)", round(Sq[-1], 2) == s and abs(q[0] * q[1] - 1) < 1e-12)
edge = lambda eta, rho: (-1 + np.sqrt(1 + 8 * rho / eta)) / (2 * rho)
check("SAM edge eta lambda (1 + rho lambda) = 2: 5.28 / 4.57 / 3.79 for rho 0.05 / 0.1 / 0.2 at eta 0.3",
      [round(edge(.3, r), 2) for r in (.05, .1, .2)] == [5.28, 4.57, 3.79] and all(np.isclose(.3 * edge(.3, r) * (1 + r * edge(.3, r)), 2) for r in (.05, .1, .2)))
check("SAM edge, notes example eta 0.5, rho 0.2: 2.62", round(edge(.5, .2), 2) == 2.62)
check("unnormalized SAM on the quadratic: x' = x - eta (H + rho H^2) x", np.allclose((lambda H, x: x - .3 * (H @ (x + .1 * H @ x)))(np.diag([3., 1]), np.array([1., 1])), np.array([1., 1]) - .3 * (np.diag([3., 1]) + .1 * np.diag([9., 1])) @ np.array([1., 1])))
q, Sq = run_uv([2.2, .1], .3, mode="norm", rho=.1)
print(f"  normalized SAM rho 0.1: end {q.round(3)}, |uv - 1| = {abs(q[0] * q[1] - 1):.3f}, mean S last 200 = {Sq[-200:].mean():.2f}")
check("normalized SAM rho 0.1 drifts to near (1, 1): mean sharpness of the last 200 steps 1.94, keeps jittering (|uv - 1| = 0.085)",
      round(Sq[-200:].mean(), 2) == 1.94 and round(abs(q[0] * q[1] - 1), 3) == .085)

# ---------- 8. warmup (Kalra & Barkeshli 2024) ----------
check("warmup: sqrt(1 - beta2) = 0.0316 = 1/31.6 for beta2 = 0.999", round(np.sqrt(1 - .999), 4) == .0316 and round(1 / np.sqrt(1 - .999), 1) == 31.6)
check("warmup: Adam threshold (2 + 2 beta1)/((1 - beta1) eta) = 38/eta for beta1 = 0.9", np.isclose((2 + 2 * .9) / (1 - .9), 38))
done()
