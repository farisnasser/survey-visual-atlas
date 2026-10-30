## objective | The SAM objective (Eq. 1)
$$\min_w\ L_S^{\text{SAM}}(w)+\lambda\lVert w\rVert_2^2,\qquad L_S^{\text{SAM}}(w)=\max_{\lVert\epsilon\rVert_p\le\rho}L_S(w+\epsilon)$$
A point is scored by the worst training loss in a parameter-space ball around it (adversarial in weights, not inputs). Motivated by a PAC-Bayes bound (Theorem 1/2) in which this neighbourhood loss controls the population loss.
### One-step approximation (Eqs. 2–3)
First-order Taylor expansion of the inner problem; for $p=2$:
$$\hat\epsilon(w)=\rho\,\frac{\nabla L_S(w)}{\lVert\nabla L_S(w)\rVert_2},\qquad \nabla L_S^{\text{SAM}}(w)\approx\nabla L_S(w)\big|_{w+\hat\epsilon(w)}$$
The Hessian terms from differentiating $\hat\epsilon(w)$ are dropped. Cost: two gradients per step.
### On screen
$L=1+0.04\lVert w\rVert^2-e^{-\lVert w-c_s\rVert^2/(2\cdot0.16^2)}-1.035\,e^{-\lVert w-c_f\rVert^2/(2\cdot0.55^2)}$.
> Check script: equal depth (0.0057 vs 0.0046), $\lambda_{\max}$ 39.2 vs 3.5. $L^{\text{SAM}}$ at $\rho=0.2$: 0.557 (sharp) vs 0.073 (flat). From $(-1.36,-0.32)$ with $\rho=0.3$: GD ends sharp, descent on the exact objective ends flat, one-step SAM keeps jittering near $(-0.18,0.14)$ at $L\approx0.89$.

## basins | What the paper reports, and what the toy shows
- Reported (not reproduced): lower test error than the base optimizer on CIFAR-10/100, ImageNet and fine-tuning tasks, robustness to label noise, and lower Hessian eigenvalues for a WideResNet trained with SAM (Fig. 3).
- **m-sharpness (Sec. 4.1):** in practice the inner max is taken per subset of $m$ examples; smaller $m$ generalizes better and its sharpness correlates better with the generalization gap.
### Notes' example
$f=\tfrac a2(\theta-\theta^\star)^2$, $\rho=0.1$: the worst nearby loss is $0.01$ for $a=2$ and $0.10$ for $a=20$, although both minima have loss 0.
> Check script (41 × 41 starts): GD ends sharp from 25%. Exact objective: 25%, 23%, 17% for $\rho=0.1,0.2,0.4$. One-step SAM: 23%, 19%, 11% sharp, with 4%, 10%, 23% of starts stuck between the wells.
