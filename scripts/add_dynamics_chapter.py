"""One-off (2026-11-08): register guide chapter 6 'dynamics' (survey II-G/H/I), its glossary terms, links and the
chapter-5 hand-off."""
import json, os
G = os.path.join(os.path.dirname(__file__), "..", "guide")
rd = lambda p: open(os.path.join(G, p), encoding="utf-8").read()
def wr(p, s): open(os.path.join(G, p), "w", encoding="utf-8").write(s)

# chapter entry
g = json.load(open(os.path.join(G, "guide.json"), encoding="utf-8"))
if not any(c["id"] == "dynamics" for c in g["chapters"]):
    i = next(k for k, c in enumerate(g["chapters"]) if c["id"] == "multiplier")
    g["chapters"].insert(i + 1, {
        "id": "dynamics", "sec": "II", "n": 6, "ready": True,
        "title": "GD as a dynamical system, noise and measuring curvature",
        "question": "What does it mean that GD converges to a saddle with probability zero, what shape does minibatch noise have, and how is the sharpness of a real network measured?",
        "terms": ["fixed-point", "measure-zero", "gradient-flow", "period-doubling", "isotropic-noise", "hvp"]})
    json.dump(g, open(os.path.join(G, "guide.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# glossary terms new to the guide
gl = json.load(open(os.path.join(G, "glossary.json"), encoding="utf-8"))
gl.update({
 "fixed-point": {"term": "fixed point of a map", "ch": "dynamics",
   "def": "A point the map does not move, $G(\\theta) = \\theta$. For the GD map $G(\\theta) = \\theta - \\eta\\nabla L(\\theta)$ this happens exactly when $\\nabla L(\\theta) = 0$, so the fixed points of GD are the critical points of $L$.",
   "tex": "G(\\theta^\\star) = \\theta^\\star", "sec": "n-1-gradient-descent-as-a-map"},
 "diffeomorphism": {"term": "diffeomorphism", "ch": "dynamics",
   "def": "A smooth map with a smooth inverse. It never folds a region of positive volume onto a set of zero volume. The GD map is one when $\\eta < 1/\\ell$, because then every factor $1 - \\eta\\lambda_j$ of $\\det DG$ is positive.",
   "sec": "n-2-stable-sets-measure-zero-almost-surely"},
 "gradient-flow": {"term": "gradient flow", "ch": "dynamics",
   "def": "The limit of GD as the step size goes to zero: the differential equation $d\\theta/ds = -\\nabla L(\\theta)$, with time $s = \\eta t$. The loss decreases along it at rate $\\lVert\\nabla L\\rVert^2$.",
   "tex": "\\frac{d\\theta}{ds} = -\\nabla L(\\theta)", "sec": "n-3-gradient-flow-invariant-sets-oscillations"},
 "isotropic-noise": {"term": "isotropic noise", "ch": "dynamics",
   "def": "Noise whose covariance is a multiple of the identity, equally strong in every direction. In $d$ dimensions only a $1/d$ share of its variance lies along any one fixed direction. Real minibatch noise is anisotropic.",
   "tex": "\\Sigma = cI", "sec": "n-4-randomness-the-shape-of-minibatch-noise"},
})
json.dump(gl, open(os.path.join(G, "glossary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# term links in the chapter text (first occurrence of each)
t = rd("dynamics/text.md")
for a, b in [("**Gradient flow.** If", "**[[gradient-flow|Gradient flow]].** If"),
             ("The theorem needs $G$ to be a **diffeomorphism**", "The theorem needs $G$ to be a **[[diffeomorphism]]**"),
             ("Noise is **isotropic** if", "Noise is **[[isotropic-noise|isotropic]]** if"),
             ("This is a **Hessian-vector product** (HVP).", "This is a **[[hvp|Hessian-vector product]]** (HVP)."),
             ("is **power iteration**.", "is **[[power-method|power iteration]]**.")]:
    assert t.count(a) == 1, a
    t = t.replace(a, b)
# orbit read box: name the two charts and the readout
a = "The charts follow the gap $|x - y|$ and the residual over the steps."
assert a in t
t = t.replace(a, "The top chart follows the gap $|x - y|$ on a log scale (zero means on the diagonal). The bottom chart follows the residual $xy - 1$ (zero means on the hyperbola of minima).")
a = "in this run it persists until $K \\approx 1.237$ and then doubles into a period-four orbit."
assert a in t
t = t.replace(a, a + " In the period-two range the readout gives the two points and the two-step multiplier, which has magnitude below one: the orbit is stable.")
a = "Violet: the central difference $(\\nabla f(x + hv) - \\nabla f(x - hv))/(2h)$."
assert a in t
t = t.replace(a, a + " The dashed amber line and the white dot mark the chosen $h$ and the forward error there.")
wr("dynamics/text.md", t)

# chapter 5 now hands off to chapter 6
m = rd("multiplier/text.md")
old = "The left side of the multiplier line opens Section III. Saddles are everywhere in high dimension, so do they ever trap gradient descent? The next chapter shows that, from a random start, GD essentially never stops at a strict saddle."
assert old in m
m = m.replace(old, "The multiplier describes one step near one point. To say what happens over a whole run, from a random start, with minibatch noise, we need three more tools: GD seen as a map with stable sets, the shape of the noise, and a way to measure curvature in a network with millions of weights. The next chapter builds them, and then Section III begins.")
wr("multiplier/text.md", m)
print("ok")
