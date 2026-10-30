## batch | Large batches, sharp minimizers
### Setup (Sec. 2)
Small batch (SB): 256 samples; large batch (LB): 10% of the training set; ADAM for both, 6 networks (fully connected on MNIST and TIMIT, shallow and deep convolutional on CIFAR-10/100), 5 runs each. Both reach high training accuracy; LB generalizes worse (Table 2, reported).
### Sharpness metric (Metric 2.1, Eq. 4)
$$\phi_{x,f}(\epsilon,A)=\frac{\max_{y\in\mathcal C_\epsilon}f(x+Ay)-f(x)}{1+f(x)}\times100,\qquad\mathcal C_\epsilon=\{z:|z_i|\le\epsilon(|(A^+x)_i|+1)\}$$
LB minimizers have much larger $\phi$ (Tables 3–4, $\epsilon=10^{-3},5\cdot10^{-4}$) and the parametric plots $f(\alpha x_{LB}+(1-\alpha)x_{SB})$ show a sharp LB and a wide SB basin (Fig. 3).
### Explanation (Sec. 4)
Noise in SB gradients pushes iterates out of sharp basins toward flatter ones; above a threshold batch size the noise is too small to do so (Fig. 4).
### Toy on screen (a model of Sec. 4, not a network)
$$f(w)=\frac1n\sum_{i=1}^{64}\big[F(w)+v_i^\top w\big],\quad\sum_iv_i=0,\qquad F(w)=0.02\lVert w\rVert^2-e^{-\frac{\lVert w-(-1,0)\rVert^2}{2\cdot0.55^2}}-e^{-\frac{\lVert w-(1,0)\rVert^2}{2\cdot0.12^2}}$$
Minibatch gradient $\nabla F(w)+\frac1B\sum_{i\in\mathcal B}v_i$: unbiased, variance $\propto1/B$.
> Check script: minima $(-0.988,0)$ and $(0.999,0)$ with Hessian eigenvalues 3.3 and 69.4; metric (4) at $\epsilon=10^{-3}$: 0.042 (flat) vs 0.931 (sharp).

## threshold | Where the runs end
> Check script (400 starts, step 0.02, 20 000 steps): sharp-basin share 1.8% ($B=1$), 2.0% ($B=2$), 23–25% ($B=3,4,6$), 20–22% ($B=8,16$), 19.3% ($B=64$); flat-basin share 30% ($B=1$, most runs still wandering), 73–81% otherwise.
The page shows the mechanism; generalization (test error) has no meaning in this toy.
