## 1. Smoothness: the gradient cannot change too fast

This chapter follows Section II-C of the survey. Chapter 2 showed that the first-order model is only trustworthy nearby. To say *how* nearby, the survey assumes the gradient cannot change too quickly. A differentiable loss is **$\beta$-[[l-smooth|smooth]]** on a region if, for any two points $x, y$ in it (survey Eq. 4),

$$\|g(x) - g(y)\|_2 \le \beta\,\|x - y\|_2.$$

In words: moving a distance $r$ changes the gradient by at most $\beta r$. This is a Lipschitz condition on the *gradient*. The simpler Lipschitz condition on a function itself is easiest to see first.

:::viz lipschitz
:::

:::read How to read this visual
**What is drawn.** The curve $f(x) = 0.8\sin 2x + 0.3x$ in grey-blue. A double cone sits on the graph at $x_0$: its edges have slopes $\pm K$, and the red regions are where a $K$-Lipschitz function is forbidden to go.

**What moves.** Slide the cone along the curve with $x_0$, and change $K$.

**What to notice.** $f$ is $K$-Lipschitz exactly when, wherever the cone is placed, the graph stays out of the red. Where the graph breaks the rule, it is marked in coral. This curve's steepest slope is $\max|f'| = 1.9$, so the readout says it is globally Lipschitz only for $K \ge 1.9$. $\beta$-smoothness is the same rule applied to the slope $f'$ instead of to $f$: the slope may not change faster than $\beta$. For a twice differentiable loss, this holds wherever the Hessian satisfies $\|H(\theta)\|_2 \le \beta$, that is, wherever no eigenvalue exceeds $\beta$ in size.
:::

## 2. The descent lemma

Integrate the gradient along the segment from $x$ to $x + \Delta$ and use Eq. (4) to bound how much it can change on the way. The result is the **[[descent-lemma|descent lemma]]** (survey Eq. 5):

$$L(x + \Delta) \le L(x) + g(x)^\top\Delta + \frac{\beta}{2}\|\Delta\|_2^2.$$

The right-hand side is a parabola in $\Delta$ that touches $L$ at $x$ and stays **above** it. So it is a guaranteed upper bound, not an approximation. Now substitute the gradient-descent step $\Delta = -\eta g(x)$ (survey Eqs. 6 and 7):

$$L(x - \eta g) \le L(x) - \eta\|g\|^2 + \frac{\beta\eta^2}{2}\|g\|^2 = L(x) - \eta\left(1 - \tfrac{\beta\eta}{2}\right)\|g\|^2.$$

The bracket is positive exactly when $0 < \eta < 2/\beta$. So under the smoothness condition, any [[step-size|step size]] below $2/\beta$ cannot increase the loss. The choice $\eta = 1/\beta$ is the most convenient: the bracket is $\tfrac12$, and every step lowers the loss by at least $\|g\|^2/(2\beta)$. This is the survey's first lesson: **step size and curvature cannot be discussed separately**. The step is only safe relative to $\beta$.

:::viz upper
:::

:::read How to read this visual
**What is drawn.** The function $f(x) = 0.4\cos 2x + 0.1x^2$ (grey-blue). Its second derivative lies between $-1.4$ and $1.8$, so its true smoothness constant is $L = 1.8$ (this visual calls the constant $L$; the survey calls it $\beta$). The violet parabola is the descent-lemma bound $f(x) + f'(x)(y - x) + \tfrac{L}{2}(y - x)^2$ at the chosen point. Where the bound dips below $f$, it is drawn red. The coral dot is where a gradient step with $\eta = 1/L$ lands: the lowest point of the parabola.

**What moves.** The point $x$ and the constant $L$.

**What to try.** Lower $L$ below $1.8$ with the point near $x = 1.57$, where $f$ bends most sharply.

**What to notice.** With $L \ge 1.8$ the parabola never crosses $f$, and the readout confirms that the actual decrease is at least the guaranteed $f'(x)^2/(2L)$. With $L$ too small, the bound fails somewhere and the guarantee can fail with it. The step $\eta = 1/L$ is exactly "jump to the bottom of the upper bound".
:::

## 3. Why the step must respect the curvature

On a pure quadratic the bound is exact, and the effect of the step size can be read off directly. For $f(x) = \tfrac{a}{2}x^2$, the gradient is $ax$, so one step multiplies $x$ by $1 - \eta a$. Four regimes follow:

- $\eta < 1/a$: slow and steady;
- $\eta = 1/a$: one step lands exactly at the minimum;
- $1/a < \eta < 2/a$: bouncing, but shrinking;
- $\eta > 2/a$: bouncing and growing, so the iterates diverge.

The threshold $2/a$ is the one-dimensional version of $2/\beta$ above.

:::viz parabola
:::

:::read How to read this visual
**What is drawn.** The parabola $f(x) = \tfrac{a}{2}x^2$ and the gradient-descent iterates from $x_0 = 2$, up to 25 of them, joined by lines.

**What moves.** The step size $\eta$ and the curvature $a$.

**What to notice.** The readout prints $\eta a$ and the factor $1 - \eta a$ that multiplies $x$ at every step, together with $1/a$ and $2/a$. Cross $\eta = 1/a$ and the iterates start jumping across the minimum; cross $2/a$ and they climb the walls. Doubling the curvature $a$ halves the safe step. Chapter 4 makes this the central fact of conditioning: when different directions have different curvatures, one step size must respect the *largest* one.
:::

:::problem The next step
A real loss has many curvatures at once, one per eigenvector of the Hessian. Chapter 4 asks what a single step size does when they differ by a factor of a hundred.
:::
