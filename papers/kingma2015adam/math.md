## scales | Algorithm 1
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
$$\hat m_t=\frac{m_t}{1-\beta_1^t},\quad\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad\theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$
Defaults $\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. Sec. 2.1: the step is invariant to rescaling the gradient, and $\hat m/\sqrt{\hat v}$ acts as a signal-to-noise ratio that shrinks near an optimum ("automatic annealing").
> Check script: multiplying every gradient by 1000 leaves the steps unchanged. On the toy, late mean loss: SGD 0.018, momentum 0.010, Adam 0.008 (best step each).

## bias | Effective step size and bias correction (Secs. 2.1, 3)
- $|\Delta_t|\le\alpha\,(1-\beta_1)/\sqrt{1-\beta_2}$ in the most severe sparse case, and $|\Delta_t|\lesssim\alpha$ otherwise: $\alpha$ sets a trust region.
- $v_t=(1-\beta_2)\sum_{i\le t}\beta_2^{t-i}g_i^2$, so $\mathbb E[v_t]=\mathbb E[g_t^2](1-\beta_2^t)+\zeta$; dividing by $1-\beta_2^t$ removes the initialization bias. Without it (RMSProp with momentum) $\beta_2$ near 1 gives unstable early training (Fig. 4, VAE).
- Theorem 4.1 / Corollary 4.2 claim $O(\sqrt T)$ regret for online convex problems; this proof was later shown to be flawed (Reddi et al. 2018, the survey's [111]: Adam can fail to converge even on convex problems).
- AdaMax (Sec. 7.1): $u_t=\max(\beta_2u_{t-1},|g_t|)$, giving $|\Delta_t|\le\alpha$.
> Check script ($\beta_1=0.9$, $\beta_2=0.999$): uncorrected steps peak at $6.57\alpha$ at $t=12$; single spike after 100 / 1000 zeros: $0.98\alpha$ / $2.52\alpha$; bound $3.16\alpha$.
