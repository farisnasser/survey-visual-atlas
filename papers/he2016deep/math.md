## chain | Residual learning (Sec. 3)
Instead of fitting a desired map $\mathcal H(x)$ directly, a block fits the residual $\mathcal F(x)=\mathcal H(x)-x$:
$$y=\mathcal F(x,\{W_i\})+x\qquad(1)$$
with an identity shortcut (no parameters); a projection $W_sx$ only when dimensions change (Eq. 2). If the identity is near-optimal, the block only has to push $\mathcal F$ towards zero. The block's Jacobian is $I+\partial\mathcal F/\partial x$, so gradients have a direct path through every shortcut.
### On screen
Tied scalar chain of depth $2K$, input 1, target 2, loss $\tfrac12(f-2)^2$: plain $f=a^Kb^K$, residual $f=(1+a)^K(1+b)^K$, start $(0.02,0.03)$.
> Check script: gradient norm at the start, plain vs residual: $0.07$ vs $1.4$ (depth 2; both still train), $7.8\times10^{-8}$ vs $4.0$ (depth 6), $7.3\times10^{-30}$ vs $8.2$ (depth 20). After 2000 GD steps at depth 6 and 20 the plain loss is still 2.0; the residual loss is $<10^{-29}$.

## imagenet | The degradation problem (Secs. 1, 4.1)
- Deeper plain networks have **higher training error** than shallower ones (Fig. 1: 56 vs 20 layers on CIFAR-10; Fig. 4: 34 vs 18 layers on ImageNet), so this is an optimization difficulty, not overfitting; the authors argue it is unlikely to be vanishing gradients alone, since they use BN and the forward/backward signals stay healthy.
- Table 2 (top-1 %, 10-crop, ImageNet val): plain-18 27.94, plain-34 28.54; ResNet-18 27.88, ResNet-34 25.03.
- Up to 152 layers on ImageNet (1st place, ILSVRC 2015 classification) and a 1202-layer ResNet on CIFAR-10.
- Our toy shows only the simplest mechanism (shortcuts keep the initial map near the identity and the gradient away from zero); the survey cites the paper via Li et al.'s finding that skip connections smooth the loss landscape.
