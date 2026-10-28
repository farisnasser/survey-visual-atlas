## prediction | The Gram-matrix view
### Setup
Training set $\{(x_i,y_i)\}_{i=1}^n$ with $\lVert x_i\rVert=1$, predictions $u(k)\in\mathbb R^n$, squared loss
$$L(\theta)=\tfrac12\lVert u-y\rVert^2$$
Convex in $u$; non-convex in the weights $\theta$.
### Linearized dynamics
With $J=\partial u/\partial\theta$ and $G=JJ^\top$ (positive semidefinite, $n\times n$), one GD step gives
$$u(k+1)-y\approx\big(I-\eta\,G(k)\big)\big(u(k)-y\big)$$
If $G(k)\approx G(0)$ and $\lambda_0=\lambda_{\min}(G(0))>0$, every residual direction contracts.
### Network on screen
$$f(x)=\frac{1}{\sqrt m}\sum_{r=1}^m a_r\,\mathrm{softplus}(w_r^\top x)$$
> Offline: at $m=4096$ the residual path matches $(I-\eta G_0)^k r_0$ to 0.03%; at $m=4$ it deviates by 14%.

## rate | Theorems 5.1 and 6.1
Deep fully connected net (5.1): if the width is $m=2^{O(H)}\,\mathrm{poly}(n,1/\lambda_{\min}(K^{(H)}),\log(1/\delta))$ and $\eta=O\big(\lambda_{\min}(K^{(H)})/(n^2 2^{O(H)})\big)$, then with probability $\ge1-\delta$:
$$L(\theta(k))\le\Big(1-\frac{\eta\,\lambda_{\min}(K^{(H)})}{2}\Big)^k L(\theta(0))$$
ResNet with branch scaling $c_{\text{res}}/H$ (6.1): the same rate with width and step polynomial in $H$.
### Why width helps
Wide nets reach zero loss while each weight moves $O(1/\sqrt m)$, so $G(k)$ stays near $G(0)$ (and near the infinite-width $K^{(H)}$).
### Limits
Sufficient widths are huge; training loss only (no generalization); smooth analytic activations; full-batch GD with small steps.
> The chart uses our network's finite-width $G(0)$ in place of $K^{(H)}$.
