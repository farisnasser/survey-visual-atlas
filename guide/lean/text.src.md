## 1. Adafactor: Adam's memory problem for large matrices

This chapter follows Sections III-I, III-J and III-K of the survey. Take a layer with weight matrix $W \in \mathbb{R}^{m\times n}$. Adam keeps a second-moment value for every weight, $mn$ numbers for that statistic alone: a $1000\times1000$ matrix needs a million. **[[adafactor-factored|Adafactor]]** asks whether that matrix of squared-gradient scales can be approximated by two small vectors, one along the rows and one along the columns.

**A toy example.** Take the squared-gradient matrix

$$Q = \begin{bmatrix}4 & 8 & 12\\ 2 & 4 & 6\end{bmatrix}.$$

Its row sums are $r = (24, 12)^\top$, its column sums $c = (6, 12, 18)^\top$, and its total is $36$. A simple rank-one reconstruction is

$$\tilde Q = \frac{r\,c^\top}{\sum_i r_i} = \frac{1}{36}\begin{bmatrix}24\\12\end{bmatrix}\begin{bmatrix}6 & 12 & 18\end{bmatrix} = \begin{bmatrix}4 & 8 & 12\\ 2 & 4 & 6\end{bmatrix}.$$

It rebuilds this matrix exactly, because its rows follow a rank-one pattern. Real squared-gradient matrices are not rank one, so in practice this is an approximation, with its own normalization and stabilization details.

**What was saved?** Instead of $mn$ statistics, about $m + n$: for a $1000\times1000$ matrix, roughly $2000$ numbers instead of a million for that statistic. Practical versions add update clipping and scale-aware learning rates, and may drop the first moment to save even more, so the real saving depends on the configuration and the shapes of the tensors.

@@viz factor@@

:::key Remember III-I
Adafactor is Adam-like adaptive scaling with a memory-efficient row and column approximation to the squared-gradient statistics. It is not a Hessian estimate, and it trades precision of the scale for memory.
:::

## 2. Lion: sign-based momentum with one main state

Adam carries two moment states. **[[lion|Lion]]** goes the other way and simplifies the update itself: it keeps momentum-like history and uses only the *sign* of a blended direction (survey Eqs. 29 to 31):

$$c_t = \beta_1 m_{t-1} + (1 - \beta_1)\hat g_t, \qquad \theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta\,\operatorname{sign}(c_t), \qquad m_t = \beta_2 m_{t-1} + (1 - \beta_2)\hat g_t.$$

The two coefficients play different roles: $\beta_1$ blends for the current step, $\beta_2$ updates the stored state.

**What taking the sign does.** Suppose $c_t = (8, -0.2, 3)^\top$. Then $\operatorname{sign}(c_t) = (1, -1, 1)^\top$, and with $\eta = 0.01$ (ignoring decay) the step is $(-0.01, +0.01, -0.01)^\top$. Every nonzero coordinate moves by exactly $\eta$; only the direction differs. Gradient magnitudes still shape the accumulated direction, but not the size of the final step. This is neither ordinary gradient normalization nor Hessian preconditioning: it deliberately discards magnitude information.

@@viz lion@@

Lion was found by symbolic program search, and its original study reported gains in several settings. A compact state is not automatically better, though: the fixed step size changes how the learning rate and the weight decay must be tuned, and results depend on the task.

:::key Remember III-J
Lion uses momentum to decide the sign of each move instead of keeping Adam's separate per-coordinate scale. One state vector, but magnitudes are thrown away.
:::

## 3. Other modern designs and a synthesis

First-order design did not stop there. D-Adaptation aims to reduce manual learning-rate selection in specific theoretical settings; schedule-free optimization reduces reliance on predefined learning-rate schedules; Lookahead combines fast weights with a slowly moving copy. Variants such as RAdam, AdaBelief and AdaBound change how gradient history is used for stability or scaling. None of these needs to be treated as a separate mathematical framework: they change what is remembered, forgotten, or scaled.

@@viz lookahead@@

@@viz three@@

**The complete first-order roadmap.**

| method | what it remembers | why it exists | key limitation |
|---|---|---|---|
| GD | nothing | follow the full gradient | full-gradient cost; conditioning |
| SGD | nothing | cheaper minibatch updates | sampling noise |
| momentum | a running direction | use persistent movement | overshoot; one state vector |
| Nesterov | the previous movement | look-ahead gradient | tuning; guarantees need assumptions |
| AdaGrad | cumulative squares | per-coordinate scales | old history never fades |
| RMSProp | moving squares | scales follow recent history | noise and tuning |
| Adam | first and second moments | direction plus scaling | two states; convergence caveats |
| AMSGrad | running maximum of $v$ | control one scaling instability | not a universal guarantee |
| AdamW | Adam states, decoupled decay | separate weight shrinkage | still adaptive-scaling dependent |
| Adafactor | factored second moment | less optimizer memory | approximation and tuning |
| Lion | one momentum state | simple sign update | magnitudes discarded |

:::idea In plain English
Section III is the story of making better use of gradients without ever computing the loss's Hessian. Each optimizer decides what to remember, what to forget, what to scale, and how much memory to spend doing it.
:::

:::key What this section prepares us for
Every method so far uses gradients and their history. None describes how the gradient *changes* across directions, which is exactly the information that matters in narrow or rotated valleys. That is curvature, and Section IV turns to it.
:::

:::problem The next step
Chapter 10 begins Section IV with Newton's method: what the Hessian adds, and what it costs.
:::
