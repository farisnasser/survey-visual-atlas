## 1. Why structure matters

This chapter follows Sections V-A and V-B of the survey. A layer's weight matrix $W \in \mathbb{R}^{m\times n}$ has $mn$ numbers, but they are not an arbitrary list. Each row belongs to one output unit and each column to one input feature. Flattening $W$ to build a diagonal preconditioner loses this two-axis organization. At the other extreme, a dense preconditioner on $\mathrm{vec}(W) \in \mathbb{R}^{mn}$ has $(mn)^2$ entries: for a $1000 \times 1000$ layer that is $10^{12}$ numbers.

Structured methods look for approximations in between, and the survey separates three kinds:
- **[[kronecker|Kronecker products]]** represent a layer-wise matrix through smaller factors (K-FAC).
- **Matrix inverse roots** transform the row and column directions (Shampoo, chapter 14).
- **Orthogonalization** treats the proposed update as a matrix whose singular values can be changed (Muon, chapter 16).

All three use tensor structure, but they do *not* estimate the same mathematical object. K-FAC approximates Fisher curvature, Shampoo accumulates gradient second moments, and Muon reshapes an update's singular spectrum without estimating either.

## 2. The gradient of a layer is an outer product

Take a fully connected layer $s = Wa$. The input activation is $a \in \mathbb{R}^n$, the pre-activation is $s \in \mathbb{R}^m$, and the back-propagated error is $\delta = \partial L/\partial s \in \mathbb{R}^m$. By the chain rule, $\partial L/\partial W_{kj} = \delta_k a_j$, so the gradient is an outer product (survey Section V-B):

$$\nabla_WL = \delta a^\top.$$

Stacking its columns into one vector gives $\mathrm{vec}(\delta a^\top) = a \otimes \delta$, where $\otimes$ is the Kronecker product: the vector of all products $a_j\delta_k$. The Fisher block of the layer is the expected outer product of this gradient vector with itself. Using the identity $(a\otimes\delta)(a\otimes\delta)^\top = (aa^\top)\otimes(\delta\delta^\top)$, it is (survey Section V-B)

$$F_W = \mathbb{E}[(aa^\top)\otimes(\delta\delta^\top)].$$

## 3. K-FAC: separate the expectation

**[[kfac|K-FAC]]** makes one approximation: it moves the expectation inside the product (survey Eq. 36),

$$F_W \approx A \otimes S, \qquad A = \mathbb{E}[aa^\top], \qquad S = \mathbb{E}[\delta\delta^\top].$$

In general $\mathbb{E}[X\otimes Y] \ne \mathbb{E}[X]\otimes\mathbb{E}[Y]$, so this is an approximation, not an algebraic identity. It is exact when $a$ and $\delta$ are independent. The reward is that the inverse factorizes too (survey Section V-B):

$$(A\otimes S)^{-1} = A^{-1}\otimes S^{-1}.$$

The dense $mn\times mn$ block is replaced by an $n\times n$ and an $m\times m$ factor, which can be inverted separately, refreshed only occasionally, and stabilized by damping. Applied to the gradient matrix, the preconditioned step has the form $S^{-1}(\nabla_WL)A^{-1}$, up to orientation and damping conventions.

:::viz kron
:::

:::read How to read this visual
**What is drawn.** Three $6\times6$ colour maps for one layer with $n = 3$ inputs and $m = 2$ outputs (blue positive, red negative, one shared scale). Left: the exact Fisher block $F_W = \mathbb{E}[(aa^\top)\otimes(\delta\delta^\top)]$. Middle: K-FAC's $A\otimes S$, built from only a $3\times3$ and a $2\times2$ matrix. Right: their difference, what K-FAC throws away.

**The toy model.** $a$ and $\delta$ are jointly Gaussian, with $\delta = \rho Ba + \sqrt{1 - \rho^2}\,z$, where $z$ is independent of $a$. For Gaussians, Isserlis' theorem gives every fourth moment exactly: $\mathbb{E}[a_j\delta_ka_l\delta_r] = A_{jl}S_{kr} + C_{jk}C_{lr} + C_{jr}C_{lk}$, with $C = \mathbb{E}[a\delta^\top]$. So the difference map is exactly the cross terms in $C$. Nothing here is sampled.

**What moves.** $\rho$, how strongly the error depends on the activation.

**What to notice.** At $\rho = 0$, $a$ and $\delta$ are independent, $C = 0$, and the right map is blank: the factorization is exact. As $\rho$ grows, the cross terms appear and the relative error in the readout grows. The storage saving does not change: $(mn)^2 = 36$ numbers become $n^2 + m^2 = 13$, and for a $1000\times1000$ layer, $10^{12}$ become $2\cdot10^6$. In a real network $\delta$ depends on $a$ nonlinearly, so how good the approximation is must be measured, not assumed.
:::

:::survey Fig. 7 of the survey: how K-FAC obtains tractable layer-wise curvature from activation and backpropagated-error statistics. The Kronecker separation is a modeling approximation, not an exact identity for arbitrary data.
public/survey3/fig7.webp
:::

:::read How to read this figure
**Top row.** A layer with input $a \in \mathbb{R}^n$ (blue circles) and pre-activation $s = Wa \in \mathbb{R}^m$ (orange circles); the error $\delta$ flows back. The gradient $\nabla_WL = \delta a^\top$ is an $m\times n$ grid, and its vectorized form gives the exact Fisher block written above.

**Bottom row, drawn to scale for $m = 4$, $n = 3$.** The $mn\times mn$ Fisher block, with $(mn)^2 = 144$ entries, against K-FAC's two factors $A$ ($n\times n$) and $S$ ($m\times m$), with $n^2 + m^2 = 25$ entries. The text at the right repeats Eq. (36), the inverse identity above, and the survey's caveat that $\mathbb{E}[X\otimes Y] \ne \mathbb{E}[X]\otimes\mathbb{E}[Y]$.
:::

## 4. What decides whether K-FAC pays off

Martens and Grosse introduced K-FAC (2015) for efficient natural-gradient-style training, and Grosse and Martens extended it to convolutional layers (2016). Its speed and quality depend on:
- the layer's structure;
- how accurate the factorization is (the visual above);
- how often the factors are re-estimated and inverted;
- the damping;
- the communication needed to average factors across workers.

Its link to information geometry is real, but the matrix it approximates is the Fisher, not the raw Hessian.

:::problem The next step
K-FAC factors a Fisher block. Shampoo also keeps one matrix per side of the weight, but it builds them from the gradients themselves. Chapter 14.
:::
