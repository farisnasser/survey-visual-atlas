## gd | Gradient descent on the cubic subproblem
$$f(x)=\tfrac12x^\top Ax+b^\top x+\tfrac\rho3\lVert x\rVert^3,\qquad x_{t+1}=x_t-\eta\nabla f(x_t)=(I-\eta A-\rho\eta\lVert x_t\rVert I)x_t-\eta b\quad(9)$$
- Assumption A: $0<\eta\le\frac1{4(\beta+\rho R)}$, $\beta=\lVert A\rVert$
- Assumption B: $x_0=-r\,b/\lVert b\rVert$ with $0\le r\le R_c$
- Prop. 2.1: a global minimizer satisfies $(A+\rho\lVert x^\star\rVert I)x^\star=-b$ and $\rho\lVert x^\star\rVert\ge-\lambda_{\min}(A)$
- Lemma 2.3: $\lVert x_t\rVert$ non-decreasing, $\lVert x_t\rVert\le R$, $x_t^\top\nabla f(x_t)\le0$; Lemma 2.4: $x_t^{(i)}b^{(i)}\le0$ in the eigenbasis
- Prop. 2.5: if $b^{(1)}\ne0$ then $x_t\to x^\star$
### Hard case
$b^{(1)}=0$ keeps $x_t^{(1)}=0$ for all $t$. Theorem 3.2: run GD on $\tilde b=b+\sigma q$, $q$ uniform on the sphere; w.h.p. $f(\tilde x_t)\le f(x^\star)+(1+\delta)\epsilon$ with only a $\log d$ cost.
> Check script: for $b=(0,-0.5)$ GD stops at the saddle $(0,0.468)$; the perturbed run reaches $f^\star=-0.13447$.

## regimes | Theorem 3.1
With $b^{(1)}\ne0$ and $\epsilon>0$, $f(x_t)\le f^\star+\epsilon$ after
$$t\gtrsim\tau_{\text{grow}}(b^{(1)})+\tau_{\text{conv}}(\epsilon)\times\begin{cases}\approx1/\epsilon&\text{small }\epsilon\text{ relative to conditioning}\\ \approx1/\sqrt{\epsilon\cdot\text{gap}}&\text{intermediate}\\ \approx\log(1/\epsilon)&\text{well conditioned}\end{cases}$$
$\tau_{\text{grow}}=0$ for convex problems; otherwise it is the time $\lVert x_t\rVert$ needs to grow out of the region of negative curvature. Near a saddle with norm close to $\gamma/\rho$ the curvature $\succeq(\rho\lVert x\rVert-\gamma)I$ is only weakly negative, so GD stalls there.
> Check script ($d=1000$): iterations to $f-f^\star<10^{-8}$ grow sharply as $\lVert b\rVert$ shrinks; $\lVert x_t\rVert$ is monotone in all five runs.
