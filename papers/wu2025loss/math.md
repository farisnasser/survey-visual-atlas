## kink | Directional stationarity and escape neurons
### Model (Eq. 1–2)
$$\hat y=H\,\sigma(Wx),\qquad \sigma(z)=\alpha_+z\,\mathbb 1_{z\ge0}+\alpha_-z\,\mathbb 1_{z<0},\qquad L=\tfrac12\sum_k\lVert\hat y_k-y_k\rVert^2$$
The loss is non-differentiable exactly where some $w_i\perp x_k$.
### Directional stationarity (Def. 3.5)
$P$ is stationary if every one-sided directional derivative $D_\Delta L(P)=\lim_{t\to0^+}\frac{L(P+t\Delta)-L(P)}t\ge0$.
### Theorem 4.2
A stationary point with **no escape neuron** (Def. 4.1) is a local minimum; for scalar output the converse also holds. Corollary 4.4: otherwise some direction gives $L(P+\epsilon\Delta)-L(P)\le C\epsilon^2$, $C<0$.
### Toy model on screen
$$L(w,a)=\tfrac12\big(a\,\mathrm{ReLU}(w)-1\big)^2$$
At the origin $L(t,0)=L(0,t)=\tfrac12$ but $L(t,t)=\tfrac12-t^2+\tfrac12t^4$: second-order escape.

## plateaus | Saddle-to-saddle from vanishing initialization (Sec. 4.2)
### Fact 4.5
Under gradient flow, for every neuron $\lVert w_i\rVert^2-h_i^2$ is conserved; with vanishing initialization $\lVert w_i\rVert=|h_i|$.
### Corollary 4.6 and Fact 4.7
Non-minimum stationary points reached this way contain a neuron with $w_i=0,h_i=0$; escaping one requires growth of small **living** neurons.
### Experiment (Sec. 4.1.1)
50 neurons, $x_k=(x_k,1)$ with $x_k\in\{-1,-0.6,-0.1,0.3,0.7\}$, $y_k\in\{0.28,-0.1,0.03,0.23,-0.22\}$, init $\mathcal N(0,(5\cdot10^{-6})^2)$, GD step $10^{-3}$, 500k epochs.
> Our run: plateaus at 0.0953 → 0.0755 → 0.0697; conservation drift $7\times10^{-4}$ (relative, from the finite step).

## neurons | What the neurons do
- dead neurons: $w_i\cdot x_k\le0$ for all $k$ — zero gradient forever (the grey wedge, 3.93–5.67 rad)
- living neurons first align in direction, then grow in size during each loss drop
- some neurons stay on a kink ($w_i\perp x_5$, 2.53 rad): GD runs through the non-differentiable set
### Network embedding (Prop. 4.8)
Splitting a neuron into copies $(\gamma_l w,\beta_l a)$ with $\sum_l\beta_l\gamma_l=1$ keeps the function but preserves stationarity / minimality only under sign conditions on $\beta_l$.
