## cnc | Correlated Negative Curvature and the two algorithms
**Assumption 1 (CNC).** With $v_w$ the eigenvector of $\lambda_{\min}(\nabla^2f(w))$:
$$\exists\gamma>0\ \forall w:\quad\mathbb E\big[\langle v_w,\nabla f_z(w)\rangle^2\big]>\gamma$$
A condition on the stochastic gradients only along the escape direction, not isotropic noise.
### CNC-PGD (Alg. 1)
$w_{t+1}=w_t-\eta\nabla f(w_t)$; if $\lVert\nabla f(w_t)\rVert^2\le g_{\text{thres}}$ and $t-t_{\text{noise}}\ge t_{\text{thres}}$: one step $w_{t+1}=w_t-r\nabla f_z(w_t)$, $z\sim P$.
### CNC-SGD (Alg. 2)
$w_{t+1}=w_t-\eta\nabla f_z(w_t)$, and every $t_{\text{thres}}$ steps $w_{t+1}=w_t-r\nabla f_z(w_t)$ with $r>\eta$. Output a uniformly random iterate.
### Results (Table 1)
- Theorem 1: CNC-PGD returns an $(\epsilon,\epsilon^{2/5})$-SOSP w.p. $1-\delta$ in $O(\epsilon^{-2}\log\frac1\epsilon)$ iterations
- Theorem 2: CNC-SGD in $O(\epsilon^{-4}\log^2\frac1\epsilon)$; no explicit dimension dependence
- Lemma 4: learning half-spaces satisfies CNC, with a bound growing with the square of the negative eigenvalue
### Toy on screen
$f_i=\tfrac12x^2-\tfrac12y^2+\tfrac14y^4+a_iy+c_ix$, zero-mean $a,c$; CNC constant $=\overline{a^2}$.
> Check script (20 runs each): with CNC both methods always escape; with $a_i=0$ none ever does; GD never does.

## dim | Why it matters in high dimension
A perturbation uniform on the sphere of radius $r$ has $\mathbb E\langle\xi,v\rangle^2=r^2/d$ for any fixed unit $v$; growing that component to $O(1)$ costs $\approx\log(\sqrt d/r)/\log(1+\eta|\lambda|)$ steps. CNC asks only for a fixed $\gamma$ along $v$.
> Check script: escape steps 23 → 71 (isotropic) and 22 → 22 (CNC) as $d$ goes from 2 to 10 000.
