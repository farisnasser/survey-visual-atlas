## 1. Why do we need another assumption?

This chapter follows Section II-C of the survey. The first-order model says that moving along $-g$ is locally downhill. But what if the loss bends upward sharply right after the step? Then the linear prediction can be badly wrong. To say *how far* a step can safely go, the survey assumes the gradient cannot change too quickly. A loss is **$\beta$-[[l-smooth|smooth]]** on a region if, for any two points $x, y$ in it (survey Eq. 4),

$$\|g(x) - g(y)\|_2 \le \beta\,\|x - y\|_2.$$

Decode it: moving a distance $r$ changes the gradient by at most $\beta r$. The constant $\beta$ caps how fast the slope can turn. For $L(x) = 5x^2$, the gradient is $g(x) = 10x$, so $|g(x) - g(y)| = 10|x - y|$: this loss is $\beta$-smooth with $\beta = 10$ (and with any larger bound). For a twice differentiable loss, the condition holds wherever no Hessian eigenvalue exceeds $\beta$ in size.

The easiest way to see the idea is the simpler Lipschitz condition on a function itself.

:::viz lipschitz
:::

:::read How to read this visual
**What is drawn.** The curve $f(x) = 0.8\sin 2x + 0.3x$ in grey-blue. A double cone sits on the graph at $x_0$: its edges have slopes $\pm K$, and the red regions are where a $K$-Lipschitz function is forbidden to go.

**What moves.** Slide the cone along the curve with $x_0$, and change $K$.

**What to notice.** $f$ is $K$-Lipschitz exactly when, wherever the cone is placed, the graph stays out of the red. Where the graph breaks the rule, it is marked in coral. This curve's steepest slope is $\max|f'| = 1.9$, so the readout says it is globally Lipschitz only for $K \ge 1.9$. $\beta$-smoothness is the same rule applied to the slope $f'$ instead of to $f$: the slope may not change faster than $\beta$. For a twice differentiable loss, this holds wherever the Hessian satisfies $\|H(\theta)\|_2 \le \beta$, that is, wherever no eigenvalue exceeds $\beta$ in size.
:::

## 2. How smoothness gives the descent lemma

Move from $x$ to $x + \Delta$. Along that segment, the exact change in loss is the integral of the slope:

$$L(x + \Delta) - L(x) = \int_0^1 g(x + s\Delta)^\top\Delta\,ds.$$

Add and subtract $g(x)$ inside the integral:

$$L(x + \Delta) - L(x) = g(x)^\top\Delta + \int_0^1 \big(g(x + s\Delta) - g(x)\big)^\top\Delta\,ds.$$

Bound the last dot product with Cauchy–Schwarz, then with smoothness, $\|g(x + s\Delta) - g(x)\| \le \beta s\|\Delta\|$. The remaining integral is at most $\int_0^1 \beta s\|\Delta\|^2 ds = \tfrac{\beta}{2}\|\Delta\|^2$. That is the **[[descent-lemma|descent lemma]]** (survey Eq. 5):

$$L(x + \Delta) \le L(x) + g(x)^\top\Delta + \frac{\beta}{2}\|\Delta\|_2^2.$$

Unlike the Taylor model, this is not an approximation. It is a guaranteed upper bound, as long as the smoothness assumption holds along the step: a parabola in $\Delta$ that touches $L$ at $x$ and stays above it.

:::viz upper
:::

:::read How to read this visual
**What is drawn.** The function $f(x) = 0.4\cos 2x + 0.1x^2$ (grey-blue). Its second derivative lies between $-1.4$ and $1.8$, so its true smoothness constant is $L = 1.8$ (this visual calls the constant $L$; the survey calls it $\beta$). The violet parabola is the descent-lemma bound $f(x) + f'(x)(y - x) + \tfrac{L}{2}(y - x)^2$ at the chosen point. Where the bound dips below $f$, it is drawn red. The coral dot is where a gradient step with $\eta = 1/L$ lands: the lowest point of the parabola.

**What moves.** The point $x$ and the constant $L$.

**What to try.** Lower $L$ below $1.8$ with the point near $x = 1.57$, where $f$ bends most sharply.

**What to notice.** With $L \ge 1.8$ the parabola never crosses $f$, and the readout confirms that the actual decrease is at least the guaranteed $f'(x)^2/(2L)$. With $L$ too small, the bound fails somewhere and the guarantee can fail with it. The step $\eta = 1/L$ is exactly "jump to the bottom of the upper bound".
:::

## 3. Substitute the actual gradient descent step

Gradient descent chooses $\Delta = -\eta g$. Put it into the three terms of the bound (survey Eqs. 6 and 7):

$$L(x - \eta g) \le L(x) + g^\top(-\eta g) + \frac{\beta}{2}\|{-\eta g}\|^2 = L(x) - \eta\|g\|^2 + \frac{\beta\eta^2}{2}\|g\|^2 = L(x) - \eta\Big(1 - \frac{\beta\eta}{2}\Big)\|g\|^2.$$

**Where does $0 < \eta < 2/\beta$ come from?** We want the subtracted term to be positive whenever $g \ne 0$: $\eta(1 - \beta\eta/2) > 0$. With $\eta > 0$ that means $1 - \beta\eta/2 > 0$, i.e. $\beta\eta < 2$, i.e. $\eta < 2/\beta$. For $\beta = 10$, any $0 < \eta < 0.2$ is covered. The stricter choice $\eta \le 1/\beta$ makes the bracket at least $\tfrac12$, so the survey can say it guarantees a decrease of at least $\tfrac{\eta}{2}\|g\|_2^2$; at exactly $\eta = 1/\beta$ the guaranteed decrease is $\|g\|^2/(2\beta)$, the largest this bound allows.

:::idea In plain English
Think of following a road downhill while only knowing its slope where you stand. Smoothness limits how violently the road can bend. A small enough step stays inside the range where the downhill direction you measured still guarantees progress.
:::

On a pure quadratic the bound is exact, and the effect of the step size can be read off directly. For $f(x) = \tfrac{a}{2}x^2$, one step multiplies $x$ by $1 - \eta a$: slow below $\eta = 1/a$, one-step convergence at $\eta = 1/a$, shrinking bounces between $1/a$ and $2/a$, and divergence beyond $2/a$, the one-dimensional version of $2/\beta$.

:::viz parabola
:::

:::read How to read this visual
**What is drawn.** The parabola $f(x) = \tfrac{a}{2}x^2$ and the gradient-descent iterates from $x_0 = 2$, up to 25 of them, joined by lines.

**What moves.** The step size $\eta$ and the curvature $a$.

**What to notice.** The readout prints $\eta a$ and the factor $1 - \eta a$ that multiplies $x$ at every step, together with $1/a$ and $2/a$. Cross $\eta = 1/a$ and the iterates start jumping across the minimum; cross $2/a$ and they climb the walls. Doubling the curvature $a$ halves the safe step. Chapter 4 makes this the central fact of conditioning: when different directions have different curvatures, one step size must respect the *largest* one.
:::

:::warn A fixed learning rate has a catch
A step that is safe where the loss is flat can be far too large where it changes quickly, and a network's loss has many different curvatures at once, one per direction. Chapter 4 shows what that does to gradient descent.
:::

:::key What you should remember
Smoothness turns the local Taylor idea into a rigorous one-step bound, Eq. (7). It is the mathematical link between a safe learning rate and how sharply the loss can curve: the step is only safe relative to $\beta$.
:::

:::problem The next step
The bound uses a single curvature constant $\beta$. Chapter 4 asks what happens when the curvature is very different along different directions.
:::
