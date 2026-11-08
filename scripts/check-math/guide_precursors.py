"""Guide chapter 9 (survey Section IV-A): the SGD stability condition (Wu, Ma & E 2018; restated as Eq. (1) of
Jastrzebski et al. 2020) and the new 'sgdstab' visual.

Model: N examples, example i has curvature h_i along the top direction, f(psi) = (1/N) sum_i h_i psi^2 / 2.
SGD with a batch B of S examples drawn without replacement: psi <- (1 - eta * mean_{i in B} h_i) psi.
Mean square: E[psi_{t+1}^2] = E[(1 - eta h_B)^2] E[psi_t^2] = ((1 - eta lam)^2 + eta^2 Var(h_B)) E[psi_t^2],
with lam = mean h_i, s^2 = population variance of h_i, Var(h_B) = s^2 (N - S) / (S (N - 1)).
"""
import numpy as np
from common import check, done

print("guide/precursors")
rng = np.random.default_rng(0)

def factor(el, es, N, S):   # el = eta*lam, es = eta*s
    return (1 - el) ** 2 + es ** 2 * (N - S) / (S * (N - 1))

# 1. the batch-mean variance formula, by exact enumeration over all batches for a small N
N = 8
h = rng.normal(size=N); s2 = h.var()
from itertools import combinations
for S in [1, 2, 3, 5, 8]:
    means = [np.mean(h[list(c)]) for c in combinations(range(N), S)]
    v = np.var(means)
    check(f"Var(h_B) = s^2 (N-S)/(S(N-1)) exactly, N=8, S={S}", abs(v - s2 * (N - S) / (S * (N - 1))) < 1e-12, f"{v:.6f}")

# 2. mean-square growth factor equals E[(1 - eta h_B)^2] (Monte Carlo on the visual's N = 100)
N = 100
z = rng.normal(size=N); z = (z - z.mean()) / z.std()
lam, s, eta = 3.0, 2.0, 0.5
h = lam + s * z
for S in [1, 4, 16, 64, 100]:
    draws = np.array([np.mean(h[rng.choice(N, S, replace=False)]) for _ in range(40000)])
    mc = np.mean((1 - eta * draws) ** 2)
    ex = factor(eta * lam, eta * s, N, S)
    check(f"E(1 - eta h_B)^2 = condition (11) LHS, S={S}", abs(mc - ex) < 0.02 * ex + 2e-3, f"MC {mc:.4f} vs {ex:.4f}")

# 3. worked numbers in the text (N = 100)
el, es = 1.5, 1.0
check("full batch (S = N): factor 0.25 < 1 (stable)", abs(factor(el, es, 100, 100) - .25) < 1e-12)
f1 = factor(el, es, 100, 1)
check("S = 1: factor 1.25 > 1 (unstable)", abs(f1 - 1.25) < 1e-12, f"{f1}")
f16 = factor(el, es, 100, 16)
check("S = 16: factor 0.3030 < 1 (stable)", abs(f16 - 0.30303) < 1e-4, f"{f16:.5f}")
# largest stable eta*s at eta*lam = 1 (multiplier 0): es_max = sqrt(S(N-1)/(N-S))
for S, v in [(1, 1.0), (4, 2.0310), (16, 4.3425)]:
    es_max = np.sqrt(S * (N - 1) / (N - S))
    check(f"edge at eta*lam = 1, S={S}: eta*s_max = {v}", abs(es_max - v) < 1e-3, f"{es_max:.4f}")
# largest stable eta*lam when s = 0 is 2 (the full-batch threshold), for every S
check("s = 0: condition reduces to |1 - eta lam| <= 1, i.e. eta lam in [0, 2]", factor(2.0, 0, 100, 1) == 1 and factor(2.0001, 0, 100, 1) > 1)
# with s > 0 the largest stable eta*lam shrinks: solve (1-x)^2 + es^2 c = 1
for S in [1, 16]:
    c = (N - S) / (S * (N - 1)); es = 0.5
    x = 1 + np.sqrt(1 - es ** 2 * c)
    check(f"eta*s = 0.5, S={S}: largest stable eta*lam = {x:.4f} < 2", x < 2, f"{x:.4f}")
done()
