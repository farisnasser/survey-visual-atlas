## gains | The stochastic approximation rule (Eq. 7)
$$x_{n+1}-x_n=a_n(\alpha-y_n),\qquad \Pr[y_n\le y\mid x_n]=H(y\mid x_n),\quad M(x)=\mathbb E[Y(x)]$$
Mean-square error $b_n=\mathbb E(x_n-\theta)^2$ (Eq. 9) satisfies $b_{n+1}-b_n=-2a_n\,\mathbb E[(x_n-\theta)(M(x_n)-\alpha)]+a_n^2\,\mathbb E[(y_n-\alpha)^2]$: a pull toward the root against accumulated noise.
### Gain conditions
- (6) $\sum a_n^2<\infty$ — the noise contributions are summable.
- (26), which implies (27) $\sum a_n=\infty$ — the steps never stop moving the iterate.
- “Type $1/n$” (Eq. 30): $c'/n\le a_n\le c''/n$ satisfies both; $a_n=1/n$ is the example.
### Modern reading (this tab)
With $M=\nabla F$ and $\alpha=0$ the rule is stochastic gradient descent, $x_{n+1}=x_n-a_ng_n$, $\mathbb E[g_n\mid x_n]=\nabla F(x_n)$. Here $F(x,y)=\log\cosh x-\tfrac x2+\log\cosh y+0.3y$.

## original | Theorems 1 and 2
Assume bounded responses (4) and (5): $M(x)<\alpha$ for $x<\theta$, $M(x)>\alpha$ for $x>\theta$.
- Theorem 1: if $\{a_n\}$ is of type $1/n$ and $M$ satisfies (5′), then $b_n\to0$.
- Theorem 2: the same if $M$ is nondecreasing with $M(\theta)=\alpha$ and $M'(\theta)>0$ (33)–(35).
- Since $b_n\to0$, $x_n\to\theta$ **in probability** (the paper does not claim almost-sure convergence; that came later).
> Check script (200 runs, $M=\tanh$, $\alpha=\tfrac12$, $y=M(x)+U(-1,1)$, $x_1=-2$): $b_n$ at $n=100, 1000, 20\,000$ is 0.0077, 0.00075, $2.9\times10^{-5}$ for $a_n=2/n$; it stays near 0.053 for a constant gain 0.2 and near 0.51 for $a_n=2/n^2$.
