## manifold | Embedded saddles and invariant manifolds
### Networks with additive units (Eq. 1)
$$f(x;\theta)=\sum_{i=1}^H g_{\text{out}}\big(\phi(g_{\text{in}}(x);u_i)\big)\,v_i$$
covers fully connected, convolutional and attention layers (units = neurons, kernels, heads).
### Theorem 1 (embedded fixed points)
A fixed point of the width-$(H-1)$ network is also a fixed point of the width-$H$ network with the same input–output map (zero unit, duplicated unit with split $v$, or rescaled unit, depending on the activation).
### Theorem 3 (invariant manifolds)
E.g. identical units stay identical: $\tfrac{d}{dt}(\theta_1-\theta_2)=-\nabla_{\theta_1}L+\nabla_{\theta_2}L=0$.
### Surface on screen
Two-layer linear net, whitened inputs, target $\mathrm{diag}(s_1,s_2)$, weights $W_1=W_2=\mathrm{diag}(a_1,a_2)$:
$$L(a_1,a_2)=\tfrac12(s_1-a_1^2)^2+\tfrac12(s_2-a_2^2)^2$$
- $(0,0)$: saddle of the width-0 net, $\nabla^2L=\mathrm{diag}(-2s_1,-2s_2)$
- $(\sqrt{s_1},0)$: rank-1 map = width-1 saddle, $\nabla^2L=\mathrm{diag}(4s_1,-2s_2)$ — weak negative curvature, long plateau
- $(\pm\sqrt{s_1},\pm\sqrt{s_2})$: global minima
> Checked: full $2\times2$ linear-net GD from this init stays diagonal and matches the surface to $10^{-8}$.

## growth | Two sources of timescale separation (Sec. 5)
### Linear in $u$ (Theorem 4)
Small init ⇒ $\dot v_i\approx\Sigma_{yz}u_i$, $\dot u_i\approx\Sigma_{yz}^\top v_i$: growth rate of each data direction = its singular value $s_k$. When the top direction reaches $O(1)$ the rest are $O(\sigma^{1-s_{r+1}/s_1})$: **low-rank** first, ordered by the data.
### Quadratic in $u$ (Proposition 5)
$f=\sum_i v_iu_i^\top Z(x)u_i$ (e.g. linear self-attention). Near zero the dynamics are rich-get-richer, like
$$\dot v=v^2\ \Rightarrow\ v(t)=\frac{1}{1/v(0)-t}$$
so the unit that starts largest wins: **sparse** units, ordered by initialization.
### Fig. 2 predictions
Flat spectrum ($\theta=0$) removes plateaus in linear nets but not in quadratic ones; larger initialization weakens plateaus; width barely changes linear-net curves.
