## 1. Why structure matters in a layer

This chapter follows Sections V-A and V-B of the survey. Take a layer with three inputs and two output neurons, $z = Wx + b$ with

$$W = \begin{bmatrix} w_{11} & w_{12} & w_{13}\\ w_{21} & w_{22} & w_{23}\end{bmatrix}.$$

Each row holds the weights feeding one output; each column holds one input's connections. The six numbers are organized by how the layer computes, not an unrelated list. If the gradient is

$$G = \nabla_W L = \begin{bmatrix}2 & 4 & 1\\ 3 & 6 & 2\end{bmatrix},$$

gradient descent with $\eta = 0.1$ takes $\Delta W = -0.1\,G$, entry by entry. Notice the second column of $G$ is exactly twice the first: relationships run across several weights at once, and plain GD simply ignores them.

**Why not just Adam?** Adam gives each entry its own learning rate, but its scaling is coordinate-wise: it does not represent how a whole row or column relates to another. Chapter 4's rotated valley showed that a useful direction can involve changing several weights together.

**Why not a full Hessian?** For $W \in \mathbb{R}^{m\times n}$ there are $mn$ weights and a full curvature matrix has $(mn)^2$ entries. With $m = n = 100$, that is $10{,}000$ weights and $100{,}000{,}000$ curvature entries for one layer. Two smaller factors of sizes $m\times m$ and $n\times n$ hold only $m^2 + n^2$ entries. They cannot represent every possible matrix, but they may capture the useful part cheaply.

:::idea In plain English
Structured methods sit between per-coordinate scaling and full matrices: they use some of the interactions that diagonal methods miss, without paying for dense second-order optimization. K-FAC, Shampoo, SOAP and Muon all use the matrix shape, but they do not estimate or normalize the same object.
:::

## 2. A layer's gradient is an outer product

Take two inputs and two outputs, $s = Wa$, ignoring the bias, so $s_1 = w_{11}a_1 + w_{12}a_2$ and $s_2 = w_{21}a_1 + w_{22}a_2$. Backpropagation delivers $\delta = (\partial L/\partial s_1, \partial L/\partial s_2)^\top$. For one weight, by the chain rule,

$$\frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial s_1}\frac{\partial s_1}{\partial w_{11}} = \delta_1 a_1,$$

and the same holds for every weight, so

$$\nabla_W L = \delta\,a^\top.$$

Each weight's gradient is its input activation times the backpropagated signal at its output: backpropagation written as a matrix product. With $a = (2, 1)^\top$ and $\delta = (3, 4)^\top$,

$$G = \delta a^\top = \begin{bmatrix}3\\4\end{bmatrix}\begin{bmatrix}2 & 1\end{bmatrix} = \begin{bmatrix}6 & 3\\ 8 & 4\end{bmatrix}.$$

## 3. K-FAC: two small factors instead of one huge block

**[[kfac|K-FAC]]** targets the layer's block of the *Fisher* matrix (chapter 12), not the Hessian. Even a $2\times2$ layer has four weights and a $4\times4 = 16$-entry block; a $1000\times1000$ layer would need a trillion. But $G = \delta a^\top$ is built from just two vectors, so K-FAC treats the two sides separately.

**The input side.** From $a = (2, 1)^\top$, $aa^\top = \begin{bmatrix}4 & 2\\ 2 & 1\end{bmatrix}$: squared activations on the diagonal, co-occurrence off it. Averaged over examples, $A = \mathbb{E}[aa^\top]$.

**The output side.** From $\delta = (3, 4)^\top$, $\delta\delta^\top = \begin{bmatrix}9 & 12\\ 12 & 16\end{bmatrix}$, and $S = \mathbb{E}[\delta\delta^\top]$. (For a true Fisher interpretation, $\delta$ must come from the model's own predicted labels, as in chapter 12.)

**What is a [[kronecker|Kronecker product]]?** A way of building a big matrix: each number of $A$ becomes a block, that number times $S$,

$$A\otimes S = \begin{bmatrix} a_{11}S & a_{12}S\\ a_{21}S & a_{22}S\end{bmatrix}, \qquad \text{e.g.}\quad \begin{bmatrix}2&0\\0&3\end{bmatrix}\otimes\begin{bmatrix}4&0\\0&5\end{bmatrix} = \mathrm{diag}(8, 10, 12, 15).$$

**The approximation.** The exact block is $F_W = \mathbb{E}[(aa^\top)\otimes(\delta\delta^\top)]$. K-FAC separates the expectation (survey Eq. 36):

$$F_W \approx A\otimes S.$$

In general $\mathbb{E}[X\otimes Y] \ne \mathbb{E}[X]\otimes\mathbb{E}[Y]$, because activations and backpropagated signals can depend on each other, so this is an approximation. The payoff is that $(A\otimes S)^{-1} = A^{-1}\otimes S^{-1}$: the approximate inverse acts through the small factors, giving the preconditioned step

$$\Delta W = -\eta\,S^{-1}\,G\,A^{-1}.$$

**A complete update.** With illustrative accumulated factors $A = \mathrm{diag}(4, 1)$ and $S = \mathrm{diag}(9, 16)$, and $G = \begin{bmatrix}6 & 3\\ 8 & 4\end{bmatrix}$: first $GA^{-1} = \begin{bmatrix}1.5 & 3\\ 2 & 4\end{bmatrix}$, then

$$S^{-1}GA^{-1} = \begin{bmatrix}1/6 & 1/3\\ 1/8 & 1/4\end{bmatrix},$$

and K-FAC steps by $-\eta$ times this instead of $-\eta G$. Diagonal factors only rescale entries; real, non-diagonal factors also mix them. Real implementations add damping and refresh the factors periodically.

:::survey Fig. 7 of the survey: how K-FAC obtains tractable layer-wise curvature from activation and backpropagated-error statistics. The Kronecker separation is a modeling approximation, not an exact identity for arbitrary data.
public/survey3/fig7.webp
:::

:::read How to read this figure
**(a)** One layer with $n = 3$ inputs (blue) and $m = 4$ outputs (orange). Its gradient $\nabla_W L = \delta a^\top$ has one number per weight, $m\times n$, and row $i$ is $\delta_i$ times the activation vector $a$: an outer product, shown as a grey-scale grid.

**(b)** The two factors as colour maps, $A = \mathbb{E}[aa^\top]$ ($n\times n$) and $S = \mathbb{E}[\delta\delta^\top]$ ($m\times m$), and their Kronecker product $A\otimes S$ ($12\times12$), with heavy lines between its blocks. Each $4\times4$ block is $S$ scaled by one entry of $A$.

**What to notice.** K-FAC stores $9 + 16 = 25$ numbers instead of $144$, and the big matrix is completely determined by the two small ones. That is the saving, and also the approximation: a matrix built this way cannot represent every possible dependence between activations and errors.
:::

:::viz kron
:::

:::read How to read this visual
**What is drawn.** Three $6\times6$ colour maps for one layer with $n = 3$ inputs and $m = 2$ outputs (blue positive, red negative, one shared scale). Left: the exact Fisher block $F_W = \mathbb{E}[(aa^\top)\otimes(\delta\delta^\top)]$. Middle: K-FAC's $A\otimes S$, built from only a $3\times3$ and a $2\times2$ matrix. Right: their difference, what K-FAC throws away.

**The toy model.** $a$ and $\delta$ are jointly Gaussian, with $\delta = \rho Ba + \sqrt{1 - \rho^2}\,z$, where $z$ is independent of $a$. For Gaussians, Isserlis' theorem gives every fourth moment exactly: $\mathbb{E}[a_j\delta_ka_l\delta_r] = A_{jl}S_{kr} + C_{jk}C_{lr} + C_{jr}C_{lk}$, with $C = \mathbb{E}[a\delta^\top]$. So the difference map is exactly the cross terms in $C$. Nothing here is sampled.

**What moves.** $\rho$, how strongly the error depends on the activation.

**What to notice.** At $\rho = 0$, $a$ and $\delta$ are independent, $C = 0$, and the right map is blank: the factorization is exact. As $\rho$ grows, the cross terms appear and the relative error in the readout grows. The storage saving does not change: $(mn)^2 = 36$ numbers become $n^2 + m^2 = 13$, and for a $1000\times1000$ layer, $10^{12}$ become $2\cdot10^6$. In a real network $\delta$ depends on $a$ nonlinearly, so how good the approximation is must be measured, not assumed.
:::

:::key Remember V-B
K-FAC does not invent a different gradient. It builds a tractable, structured approximation of the matrix used to transform the gradient, losing some dependencies and requiring factor updates and numerical safeguards.
:::

:::problem The next step
K-FAC's factors come from Fisher statistics. Chapter 14's Shampoo also multiplies from the left and the right, but its factors come from accumulated gradients.
:::
