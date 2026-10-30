## fit | Convex in the measure, non-convex in the particles
Problem (1): minimize over signed measures $\mu$ on the parameter space
$$J(\mu)=R\Big(\int\phi\,d\mu\Big)+G(\mu),\qquad J_m(w,\theta)=J\Big(\frac1m\sum_{j=1}^mw_j\delta_{\theta_j}\Big)$$
$J$ is convex in $\mu$; $J_m$ is non-convex in the particles. A one-hidden-layer network is $\int\phi\,d\mu$ with $\phi(\theta)(x)=\sigma(\theta\cdot x)$. Gradient descent on all particles is, as $m\to\infty$, a Wasserstein gradient flow of $J$ (Theorem 2.6).
### Theorem 3.3 (2-homogeneous case) / Theorem 3.5 (partially 1-homogeneous)
If the initial measure's support separates the small sphere $r_aS^{d-1}$ from $r_bS^{d-1}$ (resp. $\{-r_0\}\times\Theta$ from $\{r_0\}\times\Theta$) and the Wasserstein flow converges, its limit is a **global minimizer**; and $\lim_{t,m\to\infty}J(\mu_{m,t})=\min J$ (limits can be interchanged). The theorems are qualitative: no rate, and only in the many-particle limit.
### Here
$x\sim N(0,I_2)$, target with $m_0=4$ ReLU neurons, student $f=\frac1m\sum_jw_j\,\mathrm{relu}(\theta_j\cdot x)$, square loss on 600 samples, step $0.4m$ per particle, particles start on a circle of radius 0.05.
> Check script: final loss after 5 000 steps (2 inits each): $m=5$: $3.6\times10^{-5}$, $m=10$: $3.0\times10^{-5}$ (a plateau: unchanged after 20 000 steps), $m=20$: $\le1.0\times10^{-6}$, $m=100$: $3.2\times10^{-7}$; loss of $f=0$ is 0.125.

## particles | Particle trajectories (Fig. 2) and particle complexity (Fig. 3)
Because relu is positively homogeneous, $w\,\mathrm{relu}(\theta\cdot x)=\mathrm{sign}(w)\,\mathrm{relu}(|w|\theta\cdot x)$, so each neuron is the point $u_j=|w_j|\theta_j\in\mathbb R^2$ with a sign. The paper plots exactly these paths.
- Paper, Fig. 2: failure with 5 particles, success with 10 and 100 (classical ReLU parameterization, minibatch SGD; strictly, gradient flows are not defined there because $d\Phi$ is discontinuous — Proposition 4.3 uses a smoothed "signed square" parameterization).
- Paper, Fig. 3: particle gradient descent needs only slightly more particles than $m_0$, whereas convex optimization over weights on a fixed grid of particles needs exponentially many in dimension.
> On our teacher the threshold is between 10 and 20 particles; the phenomenon (few particles can stall, a modest over-parameterization succeeds) is the same.
