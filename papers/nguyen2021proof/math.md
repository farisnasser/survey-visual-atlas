## lastlayer | Track only the last hidden layer
Network (Eq. 1): $F_0=X\in\mathbb R^{N\times n_0}$, $F_l=\sigma(F_{l-1}W_l)$ for $l<L$, $F_L=F_{L-1}W_L$, loss $\Phi=\tfrac12\lVert F_L-Y\rVert_F^2$, standard parameterization, all layers trained by GD.
With features frozen, $\nabla_{W_L}\Phi=F_{L-1}^\top(F_L-Y)$ and
$$\lVert\nabla_{W_L}\Phi\rVert_F^2\ge\sigma_{\min}(F_{L-1})^2\,\lVert F_L-Y\rVert_F^2$$
(needs $n_{L-1}\ge N$). So a full-rank last-layer feature matrix alone forces descent; the proof only has to show it stays full-rank, which follows from ReLU being 1-Lipschitz — no activation-pattern or NTK tracking.
### Here
$L=3$, widths $4\to8\to N=3\to1$, $N=3$ samples; the slice is $\tfrac12\lVert F_{L-1}w-Y\rVert^2$ over $(w_1,w_2)$, Hessian $F^\top F$ restricted to those coordinates.
> Check script (ordinary run): $\sigma_{\min}(F_{L-1})$ never drops below its initial value $0.149$ (it grows to $0.69$) while the loss goes from $0.945$ to $3\times10^{-29}$. Notes' example ($F=\mathrm{diag}(1,2)$, $w=0$, $y=(1,1)$, $\eta=0.1$): $w^+=(0.1,0.2)$, loss $0.585$ — checked.

## rate | Theorem 2.2 and its corollaries
With $n_{L-1}\ge N$, constants $C_l>0$, $\alpha_0=\sigma_{\min}(F^0_{L-1})$, $\bar\lambda_l=\lVert W^0_l\rVert_2+C_l$: if
$$\alpha_0^2\ge16\lVert X\rVert_F\max_l\tfrac{\bar\lambda_{1\to L}}{\bar\lambda_lC_l}\sqrt{2\Phi(\theta_0)}\ \ (7),\quad\alpha_0^3\ge32\lVert X\rVert_F^2\bar\lambda_L\sum_{l<L}\tfrac{\bar\lambda_{1\to L-1}^2}{\bar\lambda_l^2}\sqrt{2\Phi(\theta_0)}\ \ (8),\quad\alpha_0^2\ge16\lVert X\rVert_F^2\bar\lambda_L^2\sum_{l<L}\tfrac{\bar\lambda_{1\to L-1}^2}{\bar\lambda_l^2}\ \ (9)$$
and $\eta$ satisfies (10), then $\Phi(\theta_k)\le(1-\eta\alpha_0^2/8)^k\,\Phi(\theta_0)$ (11).
- Sec. 3.1: with $W_L=0$ and the other layers scaled by a large $\beta$, (7)–(9) hold — **width $N$ suffices** for this initialization.
- Sec. 3.2: LeCun initialization, two layers: width $\Omega(N^2)$. Corollary 3.2 (Sec. 3.3): LeCun-type initialization, deep nets, one layer of width $\tilde\Omega(N^3\max(m,n_0)^3)$, the others $\Theta(m)$ — versus $\Omega(N^8)$ for every layer in Zou & Gu (2019).
> Check script: smallest $\beta$ on a $\times1.25$ grid meeting (7)–(9) is 211.8 ($\alpha_0=6685$); (10) then allows $\eta\le1.7\times10^{-16}$; (11) holds at all 4 000 steps.
