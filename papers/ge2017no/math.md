## delta | Direction of improvement and Lemma 7
**Definition 6.** For $U,U^\star\in\mathbb R^{d\times r}$: $\Delta=U-U^\star R$, $R=\arg\min_{Z^\top Z=ZZ^\top=I}\lVert U-U^\star Z\rVert_F^2$ (align before subtracting; $U$ and $UR$ give the same $UU^\top$).
**Lemma 6.** $\lVert\Delta\Delta^\top\rVert_F^2\le2\lVert M-M^\star\rVert_F^2$ and $\sigma_r(M^\star)\lVert\Delta\rVert_F^2\le\frac1{2(\sqrt2-1)}\lVert M-M^\star\rVert_F^2$.
**Lemma 7 (main).** For $f(U)=\tfrac12(UU^\top-M^\star):\mathcal H:(UU^\top-M^\star)+Q(U)$:
$$\Delta:\nabla^2f(U):\Delta=\Delta\Delta^\top:\mathcal H:\Delta\Delta^\top-3(M-M^\star):\mathcal H:(M-M^\star)+4\langle\nabla f(U),\Delta\rangle+\big[\Delta:\nabla^2Q:\Delta-4\langle\nabla Q,\Delta\rangle\big]$$
With $\mathcal H=I$, $Q=0$ and $\nabla f(U)=0$:
$$\Delta:\nabla^2f:\Delta=\lVert\Delta\Delta^\top\rVert_F^2-3\lVert M-M^\star\rVert_F^2\le-\lVert M-M^\star\rVert_F^2$$
so every stationary point with $M\ne M^\star$ is a saddle and every local minimum satisfies $UU^\top=M^\star$.
### Toy on screen
$f(u)=\tfrac12\lVert uu^\top-M^\star\rVert_F^2$, $\nabla f=2(\lVert u\rVert^2u-M^\star u)$, $\nabla^2f=2(\lVert u\rVert^2I+2uu^\top-M^\star)$. Notes' example: $M^\star=\mathrm{diag}(1,0)$ gives $\nabla^2f(0)=\mathrm{diag}(-2,0)$, $f(0.5,0)=0.28125$.
> Check script: identity holds to $1.7\times10^{-6}$ at 400 random points; at $u=0$, $\Delta^\top\nabla^2f\Delta=-2\lVert u^\star\rVert^4=-2.05\le-\lVert M^\star\rVert_F^2$.

## balance | The three problems and the balancing term
- **Matrix sensing (Theorem 3):** $(2r,\tfrac1{20})$-RIP, objective $\frac1{2m}\sum(\langle UV^\top,A_i\rangle-b_i)^2+\tfrac18\lVert U^\top U-V^\top V\rVert_F^2$: all local minima satisfy $UV^\top=M^\star$; $(\epsilon,\Omega(\sigma_r),O(\epsilon/\sigma_r))$-strict saddle
- **Matrix completion (Theorem 4):** sampling rate $p\gtrsim\mathrm{poly}(\mu,r,\kappa)\log d/d$, incoherence regularizers: no spurious local minima w.h.p.
- **Robust PCA (Theorem 5):** sparse corruptions, a smooth upper bound of the objective: no spurious local minima
Corollary 2: simple local search (e.g. perturbed GD) solves all three in polynomial time.
### Toy on screen
$f(u,v)=\tfrac12(uv-1)^2$ versus $+\tfrac18(u^2-v^2)^2$.
> Check script: without the term the Hessian at $(2,0.5)$ has a zero eigenvalue (minimum set = hyperbola); with it $(1,1)$ is an isolated strict minimum and the origin a strict saddle.
