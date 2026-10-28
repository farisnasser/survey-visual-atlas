"""Shared helpers for the per-paper math checks.

Every visual in the Atlas draws a real function and runs a real optimizer on it.
These helpers check the hand-written gradients/Hessians against finite differences
so the JS formulas (which are copied from these scripts) are known to be right.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAILS = []


def fd_grad(f, x, h=1e-6):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def fd_hess(f, x, h=1e-4):
    x = np.asarray(x, float)
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def check_grad(name, f, g, pts, tol=1e-5):
    worst = 0.0
    for p in pts:
        a = np.asarray(g(np.asarray(p, float)), float)
        b = fd_grad(f, p)
        worst = max(worst, float(np.max(np.abs(a - b)) / (1 + np.max(np.abs(b)))))
    check(f"gradient {name}", worst < tol, f"max rel err {worst:.2e}")


def check_hess_at(name, f, x, expected, tol=1e-3):
    H = fd_hess(f, x)
    err = float(np.max(np.abs(H - np.asarray(expected, float))))
    check(f"Hessian {name}", err < tol, f"fd={np.round(H, 4).tolist()} err {err:.1e}")
    return H


def eig(H):
    return np.linalg.eigvalsh((np.asarray(H) + np.asarray(H).T) / 2)


def gd(g, x0, lr, steps, clip=None):
    x = np.asarray(x0, float).copy()
    path = [x.copy()]
    for _ in range(steps):
        x = x - lr * np.asarray(g(x))
        if clip is not None:
            x = np.clip(x, -clip, clip)
        path.append(x.copy())
    return np.array(path)


def rand_pts(n, lo, hi, dim=2, seed=0):
    r = np.random.default_rng(seed)
    return r.uniform(lo, hi, size=(n, dim))


def save_json(key, name, obj):
    d = os.path.join(ROOT, "papers", key)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    print(f"  wrote papers/{key}/{name} ({os.path.getsize(p) // 1024} KB)")


def done():
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        sys.exit(1)
    print("all checks passed")


def r(a, k=4):
    """Round nested arrays for compact JSON (absolute decimals)."""
    return np.round(np.asarray(a, float), k).tolist()


def sig(a, k=4):
    """Round to k significant digits (for quantities spanning many orders of magnitude, e.g. losses)."""
    return [float(f"{v:.{k}g}") for v in np.asarray(a, float).ravel()]
