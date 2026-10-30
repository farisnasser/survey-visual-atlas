## saddle | The α-β condition
Finite sum $f=\frac1n\sum_i f_i$, $f_i^\ast=\min f_i$, $S=\arg\min f$.
### Definition 1 (Eq. 2)
$f$ satisfies the α-β-condition ($\alpha>\beta>0$) if for every $x$ some $x_p\in\mathrm{Proj}(x,S)$ gives, for all $i$,
$$\langle\nabla f_i(x),x-x_p\rangle\ \ge\ \alpha\big(f_i(x)-f_i(x_p)\big)-\beta\big(f_i(x)-f_i^\ast\big).$$
With $\beta=0$ and $S=\{x^\ast\}$ it is quasar-convexity. Unlike PL ($\tfrac12\lVert\nabla f\rVert^2\ge\mu(f-f^\ast)$), it can hold where $\nabla f=0$ and $f>f^\ast$: at a saddle.
### Example 2 (Eq. 4)
$$f=\tfrac12(f_1+f_2),\quad f_1=1-e^{-\lVert x\rVert^2},\quad f_2=1-e^{-\lVert x-(2,2)\rVert^2}$$
satisfies Definition 1 with $\beta=\alpha-8$ for some $\alpha$ (paper). Examples 4–5: matrix factorization and a two-layer ReLU net with logistic loss and weight decay satisfy it on bounded sets.
> Check script: global minimizers $(0.000674,0.000674)$ and its mirror, $f^\ast=0.499832$; saddle $(1,1)$ with Hessian eigenvalues $-0.812$, $0.271$ and $f=0.8647$. SGD (step 0.5, 400 runs): 98.3% end within 0.05 of a global minimizer, 1.8% still on the flat rim, none at the saddle.

## condition | What Theorem 1 then guarantees
Assumption 1: each $f_i$ is $L$-smooth. Assumption 2: $\sigma^2_{\rm int}=\mathbb E_i[f^\ast-f_i^\ast]<\infty$.
### Theorem 1 (SGD, $\gamma\le\frac{\alpha-\beta}{2L}$)
$$\min_{0\le k<K}\mathbb E[f(x_k)-f^\ast]\le\frac{\mathbb E\,\mathrm{dist}(x_0,S)^2}{K\gamma(\alpha-\beta)}+\frac{2L\gamma\,\sigma^2_{\rm int}}{\alpha-\beta}+\frac{2\beta\,\sigma^2_{\rm int}}{\alpha-\beta}$$
$O(K^{-1/2})$ (with $\gamma\propto K^{-1/2}$) up to a non-vanishing ball $O(\beta\sigma^2_{\rm int})$; Theorems 2–3 give the same shape for SPSmax and NGN. The paper argues the ball must appear because the class contains local minima, and reports (Table 2, Sec. 5) that $\beta\sigma^2_{\rm int}$ shrinks with width, depth and batch size; Sec. 5 checks Eq. (2) along training trajectories (ResNets on CIFAR, LSTM, U-Net, GNN, DLRM, Transformers), with the final iterate $x_K$ standing in for $x_p$ (reported, not reproduced here).
### Required α for Example 2
With $f_i^\ast=0$ and $\beta=\alpha-8$, Eq. (2) reads $\langle\nabla f_i,x-x_p\rangle+\alpha f_i(x_p)\ge8f_i(x)$, so
$$\alpha(x)=\max_i\frac{8f_i(x)-\langle\nabla f_i(x),x-x_p\rangle}{f_i(x_p)}.$$
> Check script: $\alpha\ge8.8\times10^6$ on $[-1.5,3.5]^2$, because $f_1(x_p)=9.1\times10^{-7}$; $L=2$, $\sigma^2_{\rm int}=f^\ast$; floor $2\beta\sigma^2_{\rm int}/(\alpha-\beta)\approx1.1\times10^6$. Other splits $\alpha-\beta\in\{1,8,100\}$ give the same floor.
