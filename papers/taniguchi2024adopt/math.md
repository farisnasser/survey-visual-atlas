## toy | Why Adam can fail, and the fix
Adam: $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$, $\theta_t=\theta_{t-1}-\alpha_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. Because $v_t$ contains $g_t$, the step is not conditionally unbiased; Theorem 3.1 shows RMSprop/Adam only reach a region whose size grows with the gradient bound, unless $\beta_2$ is chosen to suit the problem.
**Fix 1 (Sec. 3.1):** normalize by $v_{t-1}$, which is independent of $g_t$ given the past. **Fix 2 (Sec. 3.2):** normalize before the momentum:
$$m_t=\beta_1m_{t-1}+(1-\beta_1)\frac{g_t}{\max\{\sqrt{v_{t-1}},\epsilon\}},\qquad\theta_t=\theta_{t-1}-\alpha_tm_t\qquad(13\text{–}14)$$
Algorithm 1 (ADOPT): $v_0=g_0\odot g_0$, then the two steps above with $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t\odot g_t$.
### Toy (Eq. 17)
$f(\theta)=\theta$ on $[-1,1]$, $f_t(\theta)=k^2\theta$ w.p. $1/k$ and $-k\theta$ otherwise ($\mathbb E\,g=1$, unbiased), solution $\theta^\star=-1$.
### Notes' example
$v_{t-1}=1$, $g_t=4$, $\beta_2=0.9$: Adam-style $v_t=2.5$, step $4/\sqrt{2.5}\approx2.53$; ADOPT uses $4/\sqrt1=4$.

## beta2 | Theorem 4.1
Under Assumptions 2.1–2.3 and 2.5 (lower bound, unbiased gradients, $L$-smooth, bounded second moment $\mathbb E\lVert g\rVert^2\le G^2$; **no** almost-sure bound on $g$), a constant step $\alpha=\Theta(1/\sqrt T)$ and any $\beta_2\in[0,1)$:
$$\min_{1\le t\le T}\Big\{\mathbb E\big[\lVert\nabla f(\theta_{t-1})\rVert^{4/3}\big]^{3/2}\Big\}=O(T^{-1/2})$$
(the appendix gives the same rate for $\alpha_t=\Theta(1/\sqrt t)$), the optimal rate for smooth non-convex stochastic optimization. Algorithm 2 clips the normalized gradient to avoid instability when some entries of $v_0$ are near zero.
- Reported: on the toy, Adam fails except at $\beta_2=0.999$ for $k=10$ and fails for $k=50$; AMSGrad is much slower at $k=50$; plus an MNIST MLP, CIFAR-10 and ImageNet (ResNet, Swin Transformer), NVAE, LLaMA fine-tuning and deep RL for continuous control.
> Check script (8 seeds, $4\times10^5$ steps): $k=10$: Adam $+0.99,+1.00,+1.00$ for $\beta_2=0.1,0.5,0.9$, $-1.00$ for $0.99,0.999$; ADOPT $\le-0.99$ for all. $k=50$: Adam $\ge+0.53$ for all $\beta_2$; AMSGrad $+0.17$ to $+0.34$; ADOPT $-0.28$ to $-0.73$ and still decreasing.
