"""One-off (2026-11-12): insert the new visuals 'narrow', 'zoo' (chapter 4) and 'memory' (chapter 17) with their
'How to read this visual' boxes. Numbers quoted here are checked in scripts/check-math/guide3_numbers.py."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "guide")

def insert_after(path, anchor, block):
    p = os.path.join(G, path); s = open(p, encoding="utf-8").read()
    assert s.count(anchor) == 1, (path, anchor[:60])
    i = s.index(anchor) + len(anchor)
    s = s[:i] + block + s[i:]; open(p, "w", encoding="utf-8").write(s); print("inserted into", path)

insert_after("conditioning/text.md", "so it is solved in one step, while $y$ is multiplied by $0.99$ per step and crawls.", """

:::viz narrow
:::

:::read How to read this visual
**What is drawn.** The valley $L = \\tfrac12(100x^2 + y^2)$ of the survey's Eq. (8) as a 3D surface (height $\\sqrt{L}$), with its minimum in green and its two eigen-directions: curvature $100$ across the valley, $1$ along it. The coral ball runs gradient descent from $(1, 1)$ with the exact gradient $(100x, y)$. The chart below plots $|x_t|$ (blue) and $|y_t|$ (amber) of the same run on a log scale; the dashed lines are the predictions $|1 - 100\\eta|^t$ and $(1 - \\eta)^t$.

**What moves.** The step size $\\eta$, from $0.002$ to $0.024$. It starts at the survey's value, $\\eta = 0.01$.

**What to try.** Run it at $\\eta = 0.01$. Then lower $\\eta$ to $0.005$, and raise it to $0.019$ and then $0.021$.

**What to notice.** At $\\eta = 0.01$ the ball falls straight onto the valley floor in a single step: the factor $1 - 100\\eta$ is exactly $0$. After that it crawls along the floor, keeping $99\\%$ of $y$ each step, and needs $459$ steps to bring $|y|$ below $0.01$. A smaller $\\eta$ makes the crawl even slower. A larger one zig-zags across the valley ($1 - 100\\eta < 0$) and, past $\\eta = 0.02$, blows up. The flat direction can never get a large step, because the steep direction sets the limit. That mismatch is the condition number $\\kappa = 100$.
:::""")

insert_after("conditioning/text.md", "Escaping saddles requires separate assumptions, and sometimes random perturbations (Section II-E; the earlier survey of this course covered them in depth).", """

The survey's Fig. 2(e)–(i) shows why these targets are needed: a zero gradient can belong to very different shapes. Test them here.

:::viz zoo
:::

:::read How to read this visual
**What is drawn.** One of five local shapes, the same ones as the bottom row of the survey's Fig. 2: a local minimum $x^2 + 0.6y^2$, a strict saddle $x^2 - y^2$, a monkey saddle $x^3 - 3xy^2$, a local maximum $-(x^2 + 0.6y^2)$, and an almost-flat shelf $0.6(1 - \\tanh 4x) + 0.15y^2$. The amber marker is the candidate point, at the origin or, on the shelf, at $(0.75, 0)$. The lines through it are the Hessian's eigen-directions, labelled with their curvature.

**What moves.** The shape, the gradient tolerance $\\epsilon$ and the allowed downward curvature $\\gamma$. "Drop the ball" runs gradient descent ($\\eta = 0.1$) from a start next to the candidate point.

**What to try.** With the default $\\epsilon = 0.05$, step through the shapes and read the three verdicts. On the shelf, lower $\\epsilon$ below $0.024$. On the strict saddle, raise $\\gamma$ above $2$.

**What to notice.** The minimum, the strict saddle, the monkey saddle and the maximum all have exactly zero gradient: every one of them is a first-order stationary point. Only the second-order test separates them, and on the monkey saddle even that is fooled, because the Hessian is zero there. The shelf has no stationary point at all, yet its gradient there is only $0.0237$, so it passes as an $\\epsilon$-stationary point for any $\\epsilon \\ge 0.024$. The balls show what the tests miss: gradient descent rolls off the strict saddle, the monkey saddle and the maximum, and on the shelf it keeps sliding with an ever smaller gradient. A small gradient certifies a flat spot, not a minimum.
:::""")

insert_after("compare/text.md", "As chapter 14 showed, $m^2 + n^2$ is not always smaller than $mn$.", """

:::viz memory
:::

:::read How to read this visual
**What is drawn.** For one weight matrix with $m$ rows and $n$ columns, the number of values each optimizer keeps between steps, on a log scale, using the counts of the survey's Table II. The colours follow the legend from left to right; the number above each bar is its count. The dashed line is $mn$, the size of the weights themselves.

**What moves.** The number of rows $m$ and columns $n$, each a power of two from $16$ to $16{,}384$.

**What to try.** Start at $4096 \\times 1024$, then make the matrix square, then very thin.

**What to notice.** Momentum, Lion and Muon keep one copy of the weights' size; Adam and Sophia keep two; Adafactor keeps only $m + n$, a tiny fraction of Adam's. K-FAC and Shampoo keep $m^2 + n^2$. That is astronomically less than the full curvature matrix of $(mn)^2$ values, but it is never less than Adam's $2mn$, because $m^2 + n^2 - 2mn = (m - n)^2 \\ge 0$. It equals Adam's count only for a square matrix, and for a thin one it is far larger. This is the survey's point in Section VI-B: structured does not mean smaller than diagonal, and these counts still leave out activations, precision and temporary buffers.
:::""")
