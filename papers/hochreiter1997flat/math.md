## flat | Flat minima
### Definition (Sec. 2)
A flat minimum is a large connected region in weight space where the error stays approximately constant. The paper uses a tolerable error $E_{\rm tol}$ and boxes in weight space whose weights all give error below $E_{\rm tol}$ and similar outputs; flatness is the box volume.
### Why flat (Appendix A)
- MDL / Bayes argument via a Gibbs-algorithm variant: generalization error splits into underfitting and overfitting error; weights inside a large flat box can be given with low precision, i.e. few bits, so the network is "simple" and expected overfitting is low
- the argument uses a prior over input/output functions rather than a Gaussian weight prior
### Flat minimum search (Sec. 3)
Minimize training error together with weight precision (a penalty that grows when the box shrinks). It needs second-order derivatives, but with an efficient second-order method (Pearlmutter; Møller) it keeps backprop's order of complexity, and it prunes units, weights and inputs automatically. Experiments with feedforward and recurrent nets and stock-market prediction, where FMS beat backprop, weight decay and optimal brain surgeon/damage (reported, not reproduced).
### Toy on screen
$$f(w)=0.02\lVert w\rVert^2-e^{-\lVert w-(-1,0)\rVert^2/(2\cdot0.55^2)}-e^{-\lVert w-(1,0)\rVert^2/(2\cdot0.12^2)}$$
> Check script: minima $(-0.988,0)$, $f=-0.9802$ (Hessian eigenvalues 3.34) and $(0.999,0)$, $f=-0.9814$ (69.4). Area of $\{f<f^\ast+\epsilon\}$: 0.198 vs 0.0094 at $\epsilon=0.1$ (ratio 20.9, about 4.4 bits less for two weights). Mean error increase under weight noise $s=0.1$: 0.033 (flat) vs 0.411 (sharp); $s=0.3$: 0.233 vs 0.864.
Later work (Dinh et al. [27]) shows such flatness measures change under reparameterization.
