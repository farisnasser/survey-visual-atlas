## warm | Warmup and the instability threshold (Secs. 3–4)
Linear warmup $\eta_t=\eta_{\text{init}}+(\eta_{\text{trgt}}-\eta_{\text{init}})\,t/T_{\text{wrm}}$. GD/SGD is locally unstable when $\eta_t>\eta_c\approx2/\lambda^H_t$ ($\lambda^H_t$ = sharpness). Beyond it a catapult (loss spike) lowers the sharpness through self-stabilization, modelled by
$$x_{t+1}=(1-\eta_t\lambda^H_t)x_t,\qquad \lambda^H_{t+1}=\lambda^H_t+\eta_t(\alpha-\beta x_t^2)\qquad(1)$$
- **C1, natural progressive sharpening ($\alpha>0$):** "head-on collision", then a persistent catapult cycle with $\eta_t\approx2/\lambda^H_t$.
- **C2, natural sharpness reduction ($\alpha<0$):** one or a few separated catapults.
- Main claim: the overwhelming benefit of warmup is to let the network tolerate a larger $\eta_{\text{trgt}}$ by moving it to flatter regions.
### On screen
$f=uv$, two examples $x=1$, labels $\pm0.02$, SGD one example per step; full loss $\tfrac12(uv)^2+\tfrac12\sigma^2$, minima on the axes with $\lambda=u^2+v^2$. Target $\eta_{\text{trgt}}\lambda_0=6$.
> Check script (20 seeds): no warmup diverges 20/20; warmup 20 steps survives 15/20, 50 or 200 steps 20/20. With 200 steps $\eta_t\lambda_t$ first exceeds 2 at step 78 and stays below 2.17 afterwards; final $\lambda=0.33<2/\eta_{\text{trgt}}=1.37$.

## sweep | Warmup enlarges the trainable range (Figs. 3–4)
Reported: across FCNs, ResNets and Transformers (SGD and Adam), the largest trainable $\eta_{\text{trgt}}$ grows with $T_{\text{wrm}}$; at fixed $\eta_{\text{trgt}}$ longer warmup helps less. Sec. 6: choose $\eta_{\text{init}}$ via a loss-catapult line search to shorten or remove warmup.
> Check script (10 seeds per cell): without warmup only $\eta_{\text{trgt}}\lambda_0<4$ trains (Lewkowycz's catapult limit); at $\eta_{\text{trgt}}\lambda_0=12$ warmups of 50–400 steps train 10/10, shorter ones 0/10.

## adam | Preconditioned sharpness and GI-Adam (Secs. 4.3, 6)
Adam is GD in $\tilde\theta=P^{1/2}\theta$, so stability is set by $\lambda^{P^{-1}H}=\lambda_{\max}(P_t^{-1}\nabla^2L)$ with
$$P_t=(1-\beta_1^t)\,\mathrm{diag}\big(\sqrt{\hat v_t}\big)+\epsilon I\qquad(9)$$
Late in training $\lambda^{P^{-1}H}$ oscillates around $(2+2\beta_1)/((1-\beta_1)\eta)$ (Cohen et al. 2022). At step 1, $\hat v_1=g_0^2$, so near-zero gradient components give $\lambda^{P^{-1}H}_0\sim10^5$–$10^6$ (reported for µP / SP).
**GI-Adam:** $v_0=g_0^2$. This lowers the initial pre-conditioned sharpness by a factor $\sqrt{1-\beta_2}$ and equals Adam with an automatic warmup $\eta_t=\eta_{\text{trgt}}\sqrt{1-\beta_2^t}$.
### Notes' example
$H=\mathrm{diag}(20,1)$, update preconditioner $\mathrm{diag}(0.01,1)$: the rescaled curvature is $\mathrm{diag}(0.2,1)$, top eigenvalue 1 (not 20).
> Check script: step-1 ratio GI/Adam $=0.0317$ vs $\sqrt{1-\beta_2}=0.0316$ ($5.05\times10^5\to1.60\times10^4$, threshold 38 000).
