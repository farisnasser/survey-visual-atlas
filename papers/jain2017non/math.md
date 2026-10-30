## gpgd | Non-convex projected gradient descent (Ch. 3)
$$z_{t+1}=x_t-\eta\nabla f(x_t),\qquad x_{t+1}=\Pi_{\mathcal C}(z_{t+1})\qquad\text{(gPGD, Algorithm 2)}$$
For sparsity $\mathcal C=\mathcal B_0(s)$ the projection keeps the $s$ largest-magnitude coordinates; for low rank it is a truncated SVD.
### Definition 3.2 (RSC / RSS over $\mathcal C$)
$$\tfrac\alpha2\lVert x-y\rVert^2\le f(y)-f(x)-\langle\nabla f(x),y-x\rangle\le\tfrac\beta2\lVert x-y\rVert^2\quad\forall x,y\in\mathcal C$$
### Theorem 3.3
If $f$ is $\alpha$-RSC and $\beta$-RSS over $\mathcal C$ with $\beta/\alpha<2$ and $\eta=1/\beta$, then $f(x_T)\le f(x^\ast)+\epsilon$ after $T=O\big(\frac{\alpha}{2\alpha-\beta}\log\frac1\epsilon\big)$ steps ($x^\ast$ the best point of $\mathcal C$). Theorem 7.4 (IHT for sparse recovery) removes the condition-number restriction by projecting onto a larger sparsity level.
### The rest of the monograph
Alternating minimization (Ch. 4), EM (Ch. 5), stochastic techniques that escape saddle points (Ch. 6: noisy gradient descent, Algorithm 6, with tensor factorization as the case study), and applications: sparse recovery, low-rank matrix recovery, robust regression, phase retrieval (Chs. 7–10).
### Toy on screen
$\mathcal C=\mathcal B_0(1)\subset\mathbb R^2$, $f(w)=\tfrac12(w-w^\ast)^\top A(w-w^\ast)$, $A=\begin{pmatrix}1&c\\c&1\end{pmatrix}$, $w^\ast=(1.2,0)$. Differences of points of $\mathcal C$ cover $\mathbb R^2$, so $\alpha,\beta$ are the eigenvalues $1\mp c$.
> Check script, start $(0,1.5)$, $\eta=1/\beta$: $c=0.3$ ($\beta/\alpha=1.86$) reaches $(1.2,0)$ with per-step ratio of $f$ about 0.05; $c=0.9$ ($\beta/\alpha=19$) stops at the fixed point $(0,1.08)$ with $f=0.137$.
