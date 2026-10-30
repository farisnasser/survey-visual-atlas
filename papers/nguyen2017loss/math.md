## rank | Feature rank and Lemma 3.5
$F_k\in\mathbb R^{N\times n_k}$: hidden representation of the $N$ training samples at layer $k$; $[F_k,\mathbf 1_N]$ adds the bias column.
**Lemma 3.5.** If $\mathrm{rank}([F_k,\mathbf1_N])=N$, $\mathrm{rank}(W_l)=n_l$ for $l\in[k+2,L]$, and $\nabla_{W_{k+1}}\Phi=0$, $\nabla_{b_{k+1}}\Phi=0$, then the point is a global minimum.
### Linear-readout picture (notes)
$$\nabla_v\tfrac12\lVert[F,\mathbf1]v-y\rVert^2=[F,\mathbf1]^\top r=0\ \text{and}\ \mathrm{rank}[F,\mathbf1]=N\ \Rightarrow\ r=0$$
$x=(-1,0,1)$, features $\tanh x,\ \tanh(2x+1)$, targets $(1,0,2)$:
$$\det[F,\mathbf1]\approx-0.98225,\qquad v\approx(3.339,-2.326,1.772)\ \text{fits exactly}$$
Lemma 4.4: parameters with deficient feature rank form a set of Lebesgue measure zero.
> Check script: the near-singular area $|\det|<e$ of the $(w,c)$ square shrinks from 10% ($e=10^{-2}$) to 1.2% ($e=10^{-4}$).

## width | Theorem 3.8
Assume analytic, strictly increasing activations (sigmoid, tanh, softplus; Lemma 3.3), distinct samples, a $C^2$ loss whose critical points are global minima. If $n_k\ge N-1$ for some hidden layer $k$, every critical point that
- is non-degenerate on a set of layers $I\subseteq\{k+1,\dots,L\}$ containing $k+1$, and
- has $W_{k+2},\dots,W_L$ of full column rank (a pyramidal tail $n_{k+1}\ge n_{k+2}\ge\dots$)

is a global minimum. Corollary 3.9: the same for non-degenerate local minima. Degenerate or low-rank suboptimal minima are **not** excluded (the paper says so). Theorem 5.3: features become linearly separable for a class-dependent loss.
> Check script ($N=8$): widths 1–2 end at $\lVert\nabla L\rVert\sim10^{-5}$ with losses 0.087–0.116 (width 1) and 0.017–0.081 (width 2): bad critical points; widths 5–10 reach loss $<10^{-23}$ on every seed; widths 3–4 on 2 and 4 of 6 seeds (the rest are still descending slowly, loss $\sim10^{-6}$ and $10^{-9}$).
