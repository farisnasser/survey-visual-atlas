"""Foret, Kleiner, Mobahi, Neyshabur 2021: sharpness-aware minimization (SAM).

Eq. (1): min_w max_{|eps|_2 <= rho} L(w + eps).  Eq. (2) with p = 2: eps_hat = rho grad / |grad|.
Eq. (3): grad L^SAM(w) ~ grad L(w)|_{w + eps_hat}  (second-order terms dropped).
Toy for the page: two Gaussian wells of (nearly) equal depth, one narrow (sharp) and one wide (flat), on a shallow bowl.
"""
from common import *

print("foret2021sharpness")
CS, SS, CF, SF, B, AF = np.array([-.55, .35]), .16, np.array([.9, -.45]), .55, .04, 1.035
LR = .01


def L(z):
    z = np.asarray(z, float)
    return 1 + B * z @ z - np.exp(-((z - CS) ** 2).sum() / (2 * SS ** 2)) - AF * np.exp(-((z - CF) ** 2).sum() / (2 * SF ** 2))


def G(z):
    z = np.asarray(z, float)
    es = np.exp(-((z - CS) ** 2).sum() / (2 * SS ** 2)); ef = AF * np.exp(-((z - CF) ** 2).sum() / (2 * SF ** 2))
    return 2 * B * z + es * (z - CS) / SS ** 2 + ef * (z - CF) / SF ** 2


check_grad("two-well loss", L, G, rand_pts(8, -1.5, 1.5))


def minimize(z0, steps=4000, lr=.01):
    z = np.array(z0, float)
    for _ in range(steps):
        z = z - lr * G(z)
    return z


ms, mf = minimize(CS), minimize(CF)
Hs, Hf = fd_hess(L, ms), fd_hess(L, mf)
check("sharp and flat minima have (nearly) the same loss", abs(L(ms) - L(mf)) < .005, f"L(sharp) = {L(ms):.4f}, L(flat) = {L(mf):.4f}")
check("sharp minimum is much sharper", eig(Hs)[-1] > 8 * eig(Hf)[-1], f"lambda_max: sharp {eig(Hs)[-1]:.2f}, flat {eig(Hf)[-1]:.2f}")


def lsam(z, rho, n=128):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return max(L(z), max(L(z + rho * np.array([np.cos(t), np.sin(t)])) for t in th))


for rho in [.1, .2]:
    check(f"L^SAM(rho = {rho}) is higher at the sharp minimum than at the flat one", lsam(ms, rho) > lsam(mf, rho) + .1, f"sharp {lsam(ms, rho):.3f} vs flat {lsam(mf, rho):.3f}")


TH = np.linspace(0, 2 * np.pi, 48, endpoint=False)
DIRS = np.stack([np.cos(TH), np.sin(TH)], 1)


def Lv(Z):
    return 1 + B * (Z ** 2).sum(-1) - np.exp(-((Z - CS) ** 2).sum(-1) / (2 * SS ** 2)) - AF * np.exp(-((Z - CF) ** 2).sum(-1) / (2 * SF ** 2))


def Gv(Z):
    es = np.exp(-((Z - CS) ** 2).sum(-1) / (2 * SS ** 2))[..., None]; ef = AF * np.exp(-((Z - CF) ** 2).sum(-1) / (2 * SF ** 2))[..., None]
    return 2 * B * Z + es * (Z - CS) / SS ** 2 + ef * (Z - CF) / SF ** 2


def step(Z, method, rho, lr=LR):
    """gd: plain GD. sam: Eqs. (2)-(3). exact: descent on max over {center} + 48 points of the rho-circle (Danskin)."""
    g = Gv(Z)
    if method == "sam":
        g = Gv(Z + rho * g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12))
    elif method == "exact":
        C_ = np.concatenate([Z[:, None, :], Z[:, None, :] + rho * DIRS[None]], 1)
        W = C_[np.arange(len(Z)), Lv(C_).argmax(1)]
        g = Gv(W)
    return Z - lr * g


def fate(Z):
    return np.where(np.linalg.norm(Z - mf, axis=1) < .35, 1, np.where(np.linalg.norm(Z - ms, axis=1) < .35, 2, 0))   # 1 flat, 2 sharp, 0 neither


xs = np.linspace(-1.6, 1.6, 41); GRID = np.array([[x, y] for y in xs for x in xs])
maps, frac = {}, {}
for method in ["gd", "sam", "exact"]:
    for rho in ([0] if method == "gd" else [.1, .2, .4]):
        Z = GRID.copy()
        for _ in range(2500):
            Z = step(Z, method, rho)
        f = fate(Z); key = f"{method}-{rho}"
        maps[key] = "".join(map(str, f.tolist())); frac[key] = [round(float((f == k).mean()), 3) for k in (1, 2, 0)]
        print(f"    {key}: flat {frac[key][0]:.2f} sharp {frac[key][1]:.2f} neither {frac[key][2]:.2f}")
check("GD: about a quarter of starts end in the sharp well", .2 < frac["gd-0"][1] < .3, f"{frac['gd-0']}")
check("exact SAM objective: sharp basin shrinks as rho grows", frac["exact-0.1"][1] > frac["exact-0.2"][1] > frac["exact-0.4"][1], f"{[frac[f'exact-{r}'][1] for r in (.1, .2, .4)]}")
check("one-step SAM (Eqs. 2-3): fewer starts end sharp than GD", frac["sam-0.4"][1] < frac["gd-0"][1], f"{frac['sam-0.4']}")
check("one-step SAM stalls between wells for some starts (neither basin)", frac["sam-0.4"][2] > .1, f"neither {frac['sam-0.4'][2]}")
# the 3D tab's start: GD -> sharp, exact-SAM descent (rho = 0.3) -> flat
START = np.array([[-1.36, -.32]])
zg, ze, zp = START.copy(), START.copy(), START.copy()
for _ in range(3000):
    zg = step(zg, "gd", 0); ze = step(ze, "exact", .3); zp = step(zp, "sam", .3)
check("3D start (-1.36, -0.32): GD ends in the sharp well", fate(zg)[0] == 2, f"{np.round(zg[0], 3)}")
check("3D start: descent on the exact SAM objective (rho = 0.3) ends in the flat well", fate(ze)[0] == 1, f"{np.round(ze[0], 3)}")
print(f"    3D start: one-step SAM (rho = 0.3) ends at {np.round(zp[0], 3)} (fate {fate(zp)[0]})")
SAM3 = int(fate(zp)[0])

# notes example: f = a/2 (theta - theta*)^2 with a = 2 (flat), 20 (sharp), rho = 0.1
for a, want in [(2, .01), (20, .10)]:
    check(f"notes example: max_|e|<=0.1 of {a}/2 e^2 = {want}", abs(a / 2 * .1 ** 2 - want) < 1e-12)
# Eq. (2): for p = 2 the dual-norm solution is rho g / |g|
g = np.array([.3, -1.2]); e = .2 * g / np.linalg.norm(g)
check("first-order inner max: eps_hat = rho g/|g| maximizes g.e on the ball", abs(g @ e - .2 * np.linalg.norm(g)) < 1e-12)
save_json("foret2021sharpness", "data.json", {"sharp": r(ms, 4), "flat": r(mf, 4), "lamS": round(float(eig(Hs)[-1]), 2), "lamF": round(float(eig(Hf)[-1]), 2), "maps": maps, "frac": frac, "sam3": SAM3})
done()
