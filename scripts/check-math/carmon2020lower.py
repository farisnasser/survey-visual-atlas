"""Carmon, Duchi, Hinder, Sidford 2020: lower bounds for finding stationary points I — the hard instance (Eq. 10).

f_T(x) = -Psi(1) Phi(x_1) + sum_{i=2}^T [Psi(-x_{i-1}) Phi(-x_i) - Psi(x_{i-1}) Phi(x_i)],
Psi(t) = 0 (t <= 1/2), exp(1 - 1/(2t - 1)^2) (t > 1/2);  Phi(t) = sqrt(e) int_{-inf}^t exp(-s^2/2) ds.
Checks: zero-chain (support of the gradient grows by at most one coordinate), Lemma 2 (|grad| > 1 unless |x_i| >= 1 for all
i <= T), Lemma 3 (f(0) - inf f <= 12T, |grad| <= 23 sqrt(T)), and GD from 0 discovers one coordinate at a time, so the number
of iterations until |grad f_T| <= 1 grows linearly in T (the source of the eps^-2 bound after rescaling).
"""
from common import *
from math import erf, sqrt, e, pi

print("carmon2020lower")
Psi = lambda t: 0.0 if t <= .5 else np.exp(1 - 1 / (2 * t - 1) ** 2)
dPsi = lambda t: 0.0 if t <= .5 else Psi(t) * 4 / (2 * t - 1) ** 3
Phi = lambda t: sqrt(e) * sqrt(2 * pi) * .5 * (1 + erf(t / sqrt(2)))
dPhi = lambda t: sqrt(e) * np.exp(-t * t / 2)


def f(x):
    s = -Psi(1) * Phi(x[0])
    for i in range(1, len(x)):
        s += Psi(-x[i - 1]) * Phi(-x[i]) - Psi(x[i - 1]) * Phi(x[i])
    return s


def g(x):
    T = len(x); G = np.zeros(T)
    G[0] = -Psi(1) * dPhi(x[0])
    for i in range(1, T):
        G[i] += -Psi(-x[i - 1]) * dPhi(-x[i]) - Psi(x[i - 1]) * dPhi(x[i])
        G[i - 1] += -dPsi(-x[i - 1]) * Phi(-x[i]) - dPsi(x[i - 1]) * Phi(x[i])
    return G


check("Phi matches its integral definition", abs(Phi(.3) - sqrt(e) * np.trapezoid(np.exp(-np.linspace(-12, .3, 200001) ** 2 / 2), np.linspace(-12, .3, 200001))) < 1e-6)
check_grad("f_T (T = 5)", f, g, rand_pts(20, -2, 2, dim=5))
check("Lemma 1.ii: Psi(x) Phi'(y) > 1 for x >= 1, |y| < 1", min(Psi(x) * dPhi(y) for x in np.linspace(1, 4, 31) for y in np.linspace(-.999, .999, 41)) > 1)
# zero-chain: if supp(x) in {1..i-1} then supp(grad) in {1..i}
R = np.random.default_rng(45); ok = True
for _ in range(300):
    T = 8; i = R.integers(1, T); x = np.zeros(T); x[:i - 1] = R.normal(0, 2, i - 1)
    ok &= np.all(g(x)[i:] == 0)
check("zero-chain: the gradient reveals at most one new coordinate", ok)
# Lemma 2
viol = 0
for _ in range(2000):
    T = 6; x = R.normal(0, 1.5, T)
    if np.any(np.abs(x) < 1) and np.linalg.norm(g(x)) <= 1:
        viol += 1
check("Lemma 2: |grad f_T(x)| > 1 whenever some |x_i| < 1", viol == 0)
T = 10
x_far = np.ones(T) * 3
inf_est = min(f(x_far), min(f(R.normal(0, 3, T)) for _ in range(2000)))
check("Lemma 3.i: f_T(0) - inf f_T <= 12 T (sampled)", f(np.zeros(T)) - inf_est <= 12 * T, f"f(0) - min sampled = {f(np.zeros(T)) - inf_est:.2f}")
check("Lemma 3.ii: |grad f_T| <= 23 sqrt(T) (sampled)", max(np.linalg.norm(g(R.normal(0, 2, T))) for _ in range(2000)) <= 23 * np.sqrt(T))


def gd_count(T, h=.05, kmax=200000):
    x = np.zeros(T); disc = []
    for k in range(kmax):
        G = g(x)
        if np.linalg.norm(G) <= 1:
            return k, disc
        nz = int(np.max(np.nonzero(x)[0]) + 1) if np.any(x) else 0
        if len(disc) < nz:
            disc.append(k)
        x = x - h * G
    return kmax, disc


counts = {T: gd_count(T)[0] for T in (1, 2, 4, 8, 12, 16, 20)}
print("  GD (step 0.05) iterations until |grad| <= 1:", counts)
Ts = np.array(list(counts)); Ks = np.array([counts[t] for t in Ts])
sl = np.polyfit(Ts[2:], Ks[2:], 1)[0]
check("iterations grow linearly with T (one coordinate per stage)", np.all(np.diff(Ks) > 0) and np.corrcoef(Ts[2:], Ks[2:])[0, 1] > .99, f"about {sl:.1f} iterations per coordinate")
K2, disc2 = gd_count(2)
# path for the 3-D tab (T = 2): x2 stays exactly 0 while x1 <= 1/2
x = np.zeros(2); first = None
for k in range(400):
    if first is None and x[1] != 0:
        first = (k, x[0])
    x = x - .05 * g(x)
check("T = 2: x_2 stays exactly 0 until x_1 passes 1/2", first is not None and first[1] > .5, f"x_2 first moves at step {first[0]} with x_1 = {first[1]:.3f}")
save_json("carmon2020lower", "data.json", {"counts": [[int(t), int(counts[t])] for t in counts], "slope": round(float(sl), 2), "first": [first[0], round(float(first[1]), 4)]})
done()
