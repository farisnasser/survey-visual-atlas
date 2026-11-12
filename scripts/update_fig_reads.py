"""One-off (2026-11-12): the survey's Figs. 2, 3, 4, 6, 10, 11 and 14 were redrawn as 3D surfaces with balls, and its
captions are final. Update the guide's :::survey caption lines and the 'How to read this figure' boxes that described
the old contour plots."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "guide")
CAP = {
    1: "Fig. 1 of the survey: organization of the survey. The branches separate Hessian curvature, information geometry, gradient-statistical preconditioning, and matrix orthogonalization; their similar algebra does not make them the same approximation.",
    2: "Fig. 2 of the survey: loss geometry in two dimensions. Gradients indicate local slope; Hessian eigenvectors indicate directions of principal curvature. The bowl and saddle are local models, not claims about global network landscapes.",
    3: "Fig. 3 of the survey: why conditioning and coordinate choice matter. The Hessian eigenvalues are unchanged by rotation, but diagonal rescaling is tied to the chosen coordinates; full-matrix transformations can follow rotated principal axes.",
    7: "Fig. 7 of the survey: how K-FAC obtains tractable layer-wise curvature from activation and backpropagated-error statistics. The Kronecker separation is a modeling approximation, not an exact identity for arbitrary data.",
    8: "Fig. 8 of the survey: matrix geometry in Muon. Orthogonalization changes the singular spectrum of the proposed weight update without estimating the model's Hessian; Newton–Schulz iterations are a numerical device for the matrix operation.",
    9: "Fig. 9 of the survey: recommended evaluation dashboard. Matching compute, data, initialization, and tuning effort is necessary before interpreting empirical optimizer comparisons.",
    10: "Fig. 10 of the survey: the analytic comparisons recommended in Section VI-F. (a) GD, heavy-ball momentum, Adam and exact Newton as balls rolling over one 3D landscape; (b, c) four updates built from one matrix gradient.",
    11: "Fig. 11 of the survey: the construction proposed in Appendix C-A, drawn from the 2 × 2 matrix of Eq. (43). Solid lines are the first steps, dotted lines the next 11.",
    14: "Fig. 14 of the survey: the saddle of Eq. (53) from (1, 0.1). GD moves away from the saddle; the unrestricted Newton step of Eq. (55) lands exactly on it; damping as in Eq. (33) makes the step escape again.",
}
READ = {
    ("taylor", 2): (
        "**What is drawn.** Four quadratics $L = \\tfrac12\\theta^\\top H\\theta$, each as contour lines",
        """**Top row.** Four quadratics $L = \\tfrac12\\theta^\\top H\\theta$ drawn as 3D surfaces: (a) a round **bowl**, $H = 2I$; (b) a **narrow valley**, $H = \\mathrm{diag}(16, 1)$; (c) the same valley **rotated** by $45^\\circ$; (d) a **saddle**, $H = \\mathrm{diag}(1, -1)$. From the white ball, the blue ball runs gradient descent with the exact gradient and the orange ball runs SGD with the same step and a noisy gradient. The dashed curves on each surface are the Hessian eigen-directions: purple where the surface curves up, red where it curves down.

**What to notice in the top row.** In the bowl both balls roll straight to the bottom. In both valleys the first steps cross the valley toward the nearest wall and zig-zag, instead of running along the floor where the minimum lies; chapter 4 shows what that does to gradient descent. The rotated valley gives gradient descent exactly the same trouble, because gradient descent only feels the eigenvalues, which rotation does not change. In the saddle the balls slide off along the red, downward-curving direction.

**Bottom row.** The stationary points that Section II-E tells apart, each with a ball at the point: (e) a local minimum, $\\nabla L = 0$ and $H \\succ 0$, a second-order stationary point; (f) a strict saddle, $\\nabla L = 0$ but $\\lambda_{\\min}(H) < 0$, first-order stationary but not second-order; (g) a monkey saddle, $\\nabla L = 0$ and $H = 0$, where the Hessian test cannot decide; (h) a local maximum, $H \\prec 0$; (i) an almost-flat shelf where $\\|\\nabla L\\| \\le \\epsilon$ although there is no stationary point at all, an $\\epsilon$-first-order stationary point. These are local models, not claims about the global landscape of a network. The zoo visual in chapter 4 lets you test each one."""),
    ("conditioning", 3): (
        "**What is drawn.** Contour lines of the $100:1$ quadratic in two coordinate systems.",
        """**What is drawn.** The $100:1$ valley as a 3D surface in two coordinate systems (height $\\sqrt{L}$, flat above $L = 28$). (a) Axis-aligned, $H = \\mathrm{diag}(100, 1)$. (b) Rotated by $45^\\circ$, the matrix of Eq. (41). Each method takes 30 steps from the white ball, and the two white balls are the same point in the valley's own axes, so both panels show the same problem. Blue: GD with $\\eta = 0.018$. Green: the exact diagonal preconditioner $D = \\mathrm{diag}(H)$ with $\\eta = 1$. Vermillion: one Newton step."""),
    ("momentum", 4): (
        "**(a)** Contour lines of $L = \\tfrac12(25x^2 + y^2)$,",
        """**(a)** The valley $L = \\tfrac12(25x^2 + y^2)$ as a 3D surface (height $\\sqrt{L}$), with 40 steps of SGD (blue) and heavy ball with $\\mu = 0.8$ (orange), both with $\\eta = 0.07$ and the same Gaussian gradient noise, from the white ball. SGD zig-zags across the steep walls and creeps along the floor. The heavy ball travels along the valley much faster, but swings past the bottom.

**(b)** The one-dimensional recurrence above with $\\eta\\lambda = 0.3$, for $\\mu = 0, 0.6, 0.9, 0.97$. Without momentum, $\\theta_t$ decays smoothly. With more momentum, it oscillates, and at $\\mu = 0.97$ the oscillation lasts for the whole window.

**(c)** Where it settles: over the plane of $\\eta\\lambda$ and $\\mu$, green where every root of the characteristic polynomial above has size below one, so $\\theta_t$ settles, and red where one is larger, so $\\theta_t$ blows up. The black line $\\eta\\lambda = 2(1 + \\mu)$ is the boundary, the survey's stability condition."""),
    ("newton", 6): (
        "**(c)** An ill-conditioned bowl $\\tfrac12(10x^2 + y^2)$ and one point.",
        """**(c)** An ill-conditioned bowl $\\tfrac12(10x^2 + y^2)$ drawn in 3D, and one white ball. One GD step $-\\eta g$ with $\\eta = 0.09$ (blue) runs mostly across the valley. The Newton step $-H^{-1}g$ (vermillion) goes straight to the bottom."""),
    ("intro", 10): (
        "**Panel (a).** Contour lines of the curved valley",
        """**Panel (a).** The curved valley $L = (1-x)^2 + 10(y - x^2)^2$ as a 3D surface (height $\\log(1 + L)$), with its minimum $(1, 1)$ marked by the yellow star. All four methods start from the white ball at $(-1.2, 1)$. Gradient descent (blue, $\\eta = 0.01$), heavy-ball momentum (orange) and Adam (green) follow the curved floor for 300 steps each; exact Newton (vermillion) reaches the minimum in 8 large steps, using the Hessian. Chapter 11 runs this race live."""),
    ("conditioning", 11): (
        "**What is drawn.** Level sets of the rotated quadratic of Eq. (43)",
        """**What is drawn.** The rotated quadratic of Eq. (43) as a 3D surface (height $\\sqrt{L}$, flat above $L = 27$). From the white ball $\\theta_0 = (1, 0)$, where $L = 25.25$, the solid lines are the first steps of the three methods: GD with $\\eta = 0.01$ (blue) to $(0.495, 0.495)$ with $L = 0.245$; the exact diagonal step $D = 50.5I$ (green) to $(0, 0.9802)$ with $L \\approx 24.26$; Newton (vermillion) to $(0, 0)$ with $L = 0$. The dotted lines are the next 11 steps of GD and of the diagonal method."""),
    ("newton", 14): (
        "Contours of $L = \\tfrac12x^2 - y^2$",
        """The strict saddle $L = \\tfrac12x^2 - y^2$ as a 3D surface. From the white ball $(1, 0.1)$: GD with $\\eta = 0.1$ (blue, 10 steps) contracts $x$ by $0.9$ and grows $y$ by $1.2$ per step, so it rolls off the saddle along $y$. The Newton step of Eq. (55) (vermillion) lands exactly on the saddle, marked with a cross. Damping with $\\lambda = 3$ (purple, 2 steps) makes $H + 3I = \\mathrm{diag}(4, 1)$ positive definite; its step triples $y$ and escapes faster than GD."""),
}
for d in os.listdir(G):
    p = os.path.join(G, d, "text.md")
    if not os.path.exists(p):
        continue
    L = open(p, encoding="utf-8").read().split("\n"); out = []; i = 0; changed = False
    while i < len(L):
        l = L[i]
        if l.startswith(":::survey Fig. "):
            n = int(l.split("Fig. ")[1].split(" ")[0])
            if n in CAP and l != ":::survey " + CAP[n]:
                l = ":::survey " + CAP[n]; changed = True
            out.append(l); i += 1
            key = (d, n)
            if key in READ:
                start, new = READ[key]
                # copy up to the read box, then replace the paragraph(s) of the box that describe the drawing
                while not L[i].startswith(":::read"):
                    out.append(L[i]); i += 1
                out.append(L[i]); i += 1
                j = i
                while not L[j].startswith(start):
                    assert not L[j].startswith(":::") or L[j].startswith(":::read"), (d, n, "start not found")
                    j += 1
                out += L[i:j]
                # what to drop: for momentum/taylor/intro the replaced block runs to the next paragraph not covered
                stop_prefixes = {"taylor": ":::", "momentum": ":::", "intro": "**Panel (b)", "conditioning": "**What to notice", "newton": ":::"}
                k = j + 1
                if d == "newton" and n == 6:
                    k = j + 1   # (c) is one paragraph
                else:
                    while not L[k].startswith(stop_prefixes[d]):
                        k += 1
                    if d in ("conditioning", "intro"):
                        # keep the blank line before the next kept paragraph
                        while L[k - 1] == "":
                            k -= 1
                out += new.split("\n")
                if d == "newton" and n == 6:
                    pass
                i = k; changed = True
            continue
        out.append(l); i += 1
    if changed:
        open(p, "w", encoding="utf-8").write("\n".join(out)); print("updated", d)
