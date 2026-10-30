## slice | Setting (Assumptions 3.1–3.2, Algorithm 1)
Data: $\lVert x_i\rVert_2=1$ and $(x_i)_d=\mu>0$ (3.1); $\lVert x_i-x_j\rVert_2\ge\phi>0$ (3.2). Square loss of an $L$-hidden-layer ReLU network; hidden weights $W_l$ with rows $\sim N(0,\tfrac2mI)$, output rows $\sim N(0,I/d)$; (S)GD on the hidden layers.
### Here
One hidden layer ($L=1$), $f(x)=\sum_{r=1}^mv_r\,\mathrm{relu}(w_r\cdot x)$, $n=10$, $d=10$, $\mu=0.5$, random $\pm1$ labels ($\phi=0.52$), step $\eta=1/\lambda_{\max}(H)$ with
$$H_{ij}=\sum_rv_r^2\,\mathbb 1[w_r\cdot x_i>0]\,\mathbb 1[w_r\cdot x_j>0]\,x_i\cdot x_j .$$
The surface is $L(W_0+aD_1+bD_2)$ with $D_1=W_T-W_0$ (the real GD displacement) and a random $D_2\perp D_1$, $\lVert D_2\rVert=\lVert D_1\rVert$.
> Check script: along the training direction ($b=0$) the slice is convex for $m=8192$ (smallest second difference $+0.014$) and not for $m=32$ ($-0.11$).

## stay | The two proof ingredients
- **Gradient lower bound** (Lemma 4.1): for all $W$ within $\tau=O(\phi^{3/2}n^{-3}L^{-6}\log^{-3/2}m)$ of $W_0$, $\lVert\nabla L(W)\rVert_F^2\ge\Omega\big(m\phi\,L(W)/(kn^2)\big)$ — so GD decreases the loss geometrically (the paper's $L$ is the average loss).
- **Trajectory length** (Eq. 4.1, Lemma 4.2): $\lVert W^{(t+1)}_l-W^{(t)}_l\rVert_2\le\sqrt{Ckn^2/(m\phi)}\,\big(\sqrt{L(W^{(t)})}-\sqrt{L(W^{(t+1)})}\big)$ telescopes, so $\lVert W^{(t)}_l-W^{(0)}_l\rVert_2\le O\big(\sqrt{kn^2\log n/(m\phi)}\big)$ for all $t$ — shrinking like $1/\sqrt m$ and staying inside the region where Lemma 4.1 and the ReLU semi-smoothness (Lemma 4.4) hold.
### Theorem 3.3 (GD)
If $m=\Omega(kn^8L^{12}\log^3m/\phi^4)$ and $\eta=O(k/(L^2m))$, then w.p. $1-O(n^{-1})$ GD reaches $\epsilon$ training loss in $T=O\big(n^2L^2\log(1/\epsilon)/\phi\big)$ iterations.
> Check script (3 000 GD steps): steps to reach $10^{-6}$ of the initial loss: 304, 257, 202, 149, 113 for $m=32,\dots,8192$; relative movement $\lVert W_T-W_0\rVert_2/\lVert W_0\rVert_2$: 1.83, 1.03, 0.54, 0.22, 0.10; flipped activation patterns: 13.4%, 11.6%, 7.0%, 3.7%, 2.0%.

## width | Over-parameterization conditions (Table 1)
- Allen-Zhu et al. (2018b): $m=\tilde\Omega(kn^{24}L^{12}/\phi^8)$, $T=O(n^6L^2\log(1/\epsilon)/\phi^2)$.
- This paper (Theorem 3.3): $m=\tilde\Omega(kn^8L^{12}/\phi^4)$, $T=O(n^2L^2\log(1/\epsilon)/\phi)$ — better by $\tilde\Omega(n^{16}/\phi^4)$ and $O(n^4/\phi)$ (Remark 3.4).
- SGD (Theorem 3.8): $m=\Omega(kn^{17}L^{12}\log^3m/(B^4\phi^8))$, $T=O(n^5\log m\log^2(1/\epsilon)/(B\phi^2))$. Two-layer specializations: Corollary 3.5 (GD) and Theorem 3.10 (SGD).
