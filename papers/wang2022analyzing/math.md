## sign | The output-layer norm (Sec. 3.2, Eq. 2)
Residual $D=F-Y$, loss $L=\lVert D\rVert^2/n$, Gram matrix $M=\tfrac2n\,\tfrac{\partial F}{\partial\theta}\tfrac{\partial F}{\partial\theta}^\top$ (same nonzero spectrum as the Gauss–Newton part of the Hessian). For $f(x)=A^\top h(W,x)$, $F$ is linear in $A$, so under gradient flow
$$\frac{d\lVert A\rVert^2}{dt}=-2\frac{\partial L}{\partial A}^\top A=-\frac4n\,D^\top F\qquad(2)$$
$\lVert A\rVert^2$ grows iff $D^\top F<0$. Empirically $M\approx M_W$ (the $W$ part), which scales with $\lVert A\rVert^2$, so $\lVert A\rVert^2$ and the sharpness move together most of the time.
### On screen
Scalar case $L(a,w)=(aw-1)^2$; $D\cdot F=(aw-1)\,aw<0$ exactly for $0<aw<1$.
> Check script: Eq. (2) holds to $10^{-4}$ by finite differences; on the valley the top Hessian eigenvalue is $2(a^2+w^2)$.

## trace | Four phases and Theorems 1–2
- **I** $\Lambda<2/\eta$: progressive sharpening ($D^\top F<0$, $\lVert A\rVert^2\uparrow$).
- **II** $\Lambda>2/\eta$: $|D^\top v_1|$ grows geometrically (Lemma 3.2).
- **III** $\Lambda>2/\eta$, falling: large $\lVert D\rVert$ makes $D^\top F>0$ (Prop. 3.1), $\lVert A\rVert^2\downarrow$; loss may rise.
- **IV** $\Lambda<2/\eta$: $|D^\top v_1|$ decays, loss drops, back to I.
**Sec. 4 model:** $f(x)=\tfrac1{\sqrt m}A^\top Wx$, labels $\pm1$. **Theorem 1:** under spectral (Assumptions 4.1–4.2), width and sample-size conditions, sharpness increases step by step until $\lVert D\rVert^2=O(\epsilon^2)$ or $\Lambda>1/\eta$. **Theorem 2:** under Assumptions 4.3–4.5, if $\Lambda(t_0)>\tfrac2\eta(1+\delta)$ it later drops below $\tfrac2\eta(1+\delta)$; above it $|D^\top v_1|$ grows by a factor $1+c_7$ per step; the non-top residual tracks a converging sequence. Sec. 5: high-frequency sharpness oscillations are not explained.
> Check script ($n=32$, $d=48$, $m=64$, $\eta=1.973$): sharpness $0.49\to$ peak 1.58 at step 17, 76 excursions above $2/\eta=1.014$ until step 225, final 1.006; $\lVert A\rVert^2$ and sharpness agree in direction in 91% of sliding 10-step windows over the first 300 steps (98% over the whole run, converged tail included); $\Delta\lVert A\rVert^2$ has the sign of $-D^\top F$ at 100% of steps.
