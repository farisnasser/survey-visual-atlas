"""Guide chapter 5 (survey Section II-F): the multiplier 1 - eta*lambda of the linearized GD map.

Checks every number in guide/multiplier/text.md and the readouts of the new 'multiplier' visual.
"""
import numpy as np
from common import check, done

print("guide/multiplier")

m = lambda el: 1 - el
# regime boundaries
check("eta*lambda = 0 gives multiplier 1 (flat direction)", m(0) == 1)
check("eta*lambda = 2 gives multiplier -1 (edge)", m(2) == -1)
check("eta*lambda = 1 gives multiplier 0 (one-step convergence)", m(1) == 0)

# worked numbers used in the text
el = -0.1; r = abs(m(el)) ** 20
check("saddle: eta*lambda = -0.1 -> multiplier 1.1, x20 = 6.73", abs(m(el) - 1.1) < 1e-12 and abs(r - 6.7275) < 1e-3, f"{r:.4f}")
el = 0.5; r = abs(m(el)) ** 20
check("stable: eta*lambda = 0.5 -> multiplier 0.5, x20 = 9.54e-7", abs(r - 9.5367e-7) < 1e-10, f"{r:.4e}")
el = 1.5; r = abs(m(el)) ** 20
check("bouncing but stable: eta*lambda = 1.5 -> multiplier -0.5, |x20| = 9.54e-7", abs(m(el) + .5) < 1e-12 and abs(r - 9.5367e-7) < 1e-10)
el = 2.2; r = abs(m(el)) ** 20
check("unstable: eta*lambda = 2.2 -> multiplier -1.2, x20 = 38.3", abs(m(el) + 1.2) < 1e-12 and abs(r - 38.338) < 1e-2, f"{r:.3f}")

# escape time formula (8): T = log(r/delta)/log(1 + eta|lambda|); example eta|lambda| = 0.1, r = 1
for d, T in [(1e-2, 48.3), (1e-6, 145.0), (1e-12, 289.9)]:
    t = np.log(1 / d) / np.log(1.1)
    check(f"escape time delta={d:g}, eta|lambda|=0.1 -> {T}", abs(t - T) < 0.1, f"{t:.2f}")
# every factor 100 in delta costs log(100)/log(1.1) = 48.3 steps
check("factor 100 in delta = 48.3 extra steps", abs(np.log(100) / np.log(1.1) - 48.32) < 0.01)

# the visual iterates u_{t+1} = (1 - eta*lambda) u_t exactly; check a direct simulation matches the closed form
for el in [-0.3, 0.2, 0.9, 1.7, 2.0, 2.4]:
    u = 1.0
    for t in range(30): u = m(el) * u
    check(f"simulation = closed form at eta*lambda={el}", abs(u - m(el) ** 30) < 1e-9 * max(1, abs(m(el)) ** 30))
done()
