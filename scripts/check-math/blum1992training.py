"""Blum & Rivest 1992: the 3-node network (2 hidden threshold units + 1 AND/threshold output) and the Set-Splitting reduction.

Paper's example: S = {s1, s2, s3}, C = {{s1, s2}, {s2, s3}}.  '+' at 000, 110, 011;  '-' at 100, 010, 001.
Planes from a split (S1, S2): a_i = -1 if s_i in S1 else n, plane a.x = -1/2 (same for b with S2); '+' iff a.x > -1/2 and b.x > -1/2.
"""
import itertools
from common import *

print("blum1992training")
n = 3
pos = [np.array(v, float) for v in ([0, 0, 0], [1, 1, 0], [0, 1, 1])]
neg = [np.array(v, float) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
C = [{0, 1}, {1, 2}]
valid = []
for assign in itertools.product([1, 2], repeat=3):
    S1 = {i for i in range(3) if assign[i] == 1}; S2 = set(range(3)) - S1
    a = np.array([-1. if i in S1 else n for i in range(3)]); b = np.array([-1. if i in S2 else n for i in range(3)])
    cls = lambda x: (a @ x > -.5) and (b @ x > -.5)
    ok = all(cls(x) for x in pos) and not any(cls(x) for x in neg)
    splits = all(not (c <= S1) and not (c <= S2) for c in C)
    check(f"split S1={sorted(S1)}: planes classify all 6 points iff it is a valid set-splitting", ok == splits)
    if ok:
        valid.append(sorted(S1))
check("only the split {s2} | {s1, s3} (and its swap) works", sorted(map(tuple, valid)) == [(0, 2), (1,)])
# the 6 points are not linearly separable (so one threshold unit cannot do it): exhaustive LP-free argument on a grid
grid = np.linspace(-3, 3, 25)
sep = any(all(w @ x + c > 0 for x in pos) and all(w @ x + c < 0 for x in neg)
          for w in itertools.product(grid, repeat=3) for c in grid for w in [np.array(w)])
check("no single plane separates '+' from '-' (grid search; also follows from 3 inequalities)", not sep)
# XOR lifted by the product feature is linearly separable: x1 + x2 - 2 x1 x2 = 1/2
X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], float); yx = np.array([0, 0, 1, 1])
z = X[:, 0] + X[:, 1] - 2 * X[:, 0] * X[:, 1]
check("XOR separable with the pairwise product feature", np.all((z > .5) == (yx == 1)))
sep2 = any(all((w @ x + c > 0) == bool(t) for x, t in zip(X, yx)) for w in itertools.product(grid, repeat=2) for c in grid for w in [np.array(w)])
check("XOR not separable in the raw inputs", not sep2)
done()
