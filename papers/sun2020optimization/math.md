## chain | Gradient explosion and vanishing (Sec. 4.1)
For a deep scalar linear network the loss is $F(w)=(w_1w_2\cdots w_L-1)^2$ (Eq. 13), and
$$\frac{\partial F}{\partial w_k}=2e\prod_{j\ne k}w_j,\qquad e=w_1\cdots w_L-1.$$
With all $w_j=2$ the gradient norm is about $2^{L-1}|e|$, with all $w_j=\tfrac12$ about $0.5^{L-1}|e|$: exponentially large or small in depth. The paper's one-dimensional example $F(w)=(w^7-1)^2$ (Fig. 2) is flat on $[-1+c,1-c]$ and steep outside $[-1-c,1+c]$.
### Remedies surveyed
Careful initialization (variance-preserving, orthogonal, dynamical isometry), normalization (batch, layer, weight normalization), changes of architecture (residual connections) — the "undesirable spectrum" view of Sec. 4. Later parts: SGD, adaptive and distributed methods (Sec. 5), and the global landscape (Sec. 6: bad local minima, mode connectivity, lottery tickets, infinite width).
### Toy on screen
Slice $w_1=a$, $w_2=\dots=w_L=b$: $F(a,b)=(ab^{L-1}-1)^2$, GD step 0.02.
> Check script, steps to $F<10^{-6}$ from $(0.3,0.3)$, $(0.9,0.9)$, $(1.25,1.25)$: $L=2$: 113, 66, 71; $L=7$: 412, 2, and the cliff start leaves the box at the first step.
> Check script: $\partial F/\partial w_1=2e\,s^{L-1}$ at $w_j\equiv s$ verified by finite differences ($L=20$: $-3.8\times10^{-6}$ at $s=\tfrac12$, $1.1\times10^{12}$ at $s=2$). For $(w^7-1)^2$: $|F'|\le4.44$ on $[-0.8,0.8]$ (and $<0.06$ on $[-0.4,0.4]$) but $|F'|\ge108$ on $[1.2,1.3]$. GD with step 0.02 (stable at $w=1$, where $F''=98$): from 0.3 it needs 305 steps to come within 0.01 of the minimum; from 1.3 it jumps to 5.83 and then to $2.5\times10^9$.
