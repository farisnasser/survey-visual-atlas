## factored | The linear interpolation experiment
$$\ell(\alpha)=J\big((1-\alpha)\theta_0+\alpha\theta_f\big),\qquad \ell'(\alpha)=\nabla J(\theta(\alpha))^\top(\theta_f-\theta_0)$$
$\theta_0$ the initialization, $\theta_f$ the SGD solution. A monotone $\ell$ says there is no obstacle on this one segment; it says nothing about the curved path SGD actually took or about other directions.
### Findings (ICLR 2015)
- Secs. 2–4: for fully connected (sigmoid, ReLU, maxout), adversarially trained maxout, convolutional maxout on CIFAR-10, an MP-DBM and an LSTM, $\ell$ decreases smoothly from $\theta_0$ to $\theta_f$ (Figs. 1–8); exceptions the paper itself shows: a shallow barrier just after $\theta_0$ for the convolutional maxout net (Fig. 6, zoomed) and a secondary high-error local minimum on the MP-DBM line that SGD never passes near (Fig. 7)
- Fig. 5 left: between two SGD solutions from different seeds the line has a barrier, so they are different minima on that line
- Sec. 5 / Fig. 14: deep linear regression has saddles but no bad minima; the factored model $(1-w_1w_2)^2$ reproduces the qualitative picture
### Toy on screen (Fig. 14)
$$J(w_1,w_2)=(1-w_1w_2)^2,\qquad\nabla^2J(0)=\begin{pmatrix}0&-2\\-2&0\end{pmatrix}$$
> Check script: SGD from $(0.12,0.05)$ ends at $(0.933,1.075)$, loss along the start→solution line strictly decreasing; the line from $(1,1)$ to $(-1,-1)$ peaks at 1 (the saddle value).

## mlp | A small network
2-16-16-1 tanh MLP, logistic loss, 200 points of two interleaved spirals, SGD (batch 20, step 0.3, 20 000 steps), two seeds.
> Check script: both runs reach training accuracy 1.000 ($J=0.0004$, $0.0006$). Largest rise above the running minimum on $[0,1]$: 0.111 at $\alpha=0.25$ (run 1, 17% of the drop) and 0.0011 (run 2). Barrier between the two solutions 1.83. Sweep over 12 more seeds at steps 0.05 and 0.3: monotone (rise < 1% of the drop) in 9/12 and 10/12. SGD strays up to 4.26 off the line ($\lVert\theta_f-\theta_0\rVert=18.8$).
### What this does and does not show
The straight-line test found "no obstacle" for the paper's seven models; our toy shows it is common but not universal. Later work (e.g. Lucas et al. 2021) studies when monotone interpolation fails.
