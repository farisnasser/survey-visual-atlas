## orbit | Two-step map and the 2-D model (Theorem 3, Prop. 3)
GD map $F(x)=x-\eta\nabla f(x)$. A period-2 orbit $\{a,b\}$ is a fixed point of $F^2$ with $F(a)=b\ne a$; it is stable when
$$\big|(F^2)'(a)\big|=\big|(1-\eta f''(a))(1-\eta f''(b))\big|<1$$
even though one step at the minimum expands ($\eta f''(x^\star)>2$). A quadratic cannot do this: both factors equal $1-\eta a$.
### Theorem 3 and Proposition 3
$f(x,y)=\tfrac12(xy-\mu)^2$, $\eta=K/\mu$. On $xy=\mu$ the top Hessian eigenvalue is $(x-y)^2+2\mu\ge2\mu$, so for $K>1$ no minimum is stable. If $x,y>0$ throughout and $1<K<1.5$, then $|x-y|\to0$ along the iterates with $xy>\mu$; for $1<K<\sqrt{4.5}-1$ GD converges to the period-2 orbit on the diagonal.
> Check script (start $(1.3,0.5)$): $K=1.05$ gives $|x-y|$: $0.51\to2\times10^{-15}$ and the orbit $\{0.8729,1.0911\}$ with two-step multiplier 0.595; $K=1.1$ gives $\{0.8146,1.1161\}$. More unbalanced starts such as $(1.6,0.25)$ send $y$ negative, outside the theorem.

## bif | Theorems 1–2
**Theorem 1 (1-D, local).** At a local minimum $x^\star$ with $f^{(3)}(x^\star)\ne0$ and $3[f^{(3)}]^2-f''f^{(4)}>0$, there are learning rates slightly above $2/f''(x^\star)$ for which $F^2$ has fixed points near $x^\star$. Lemma 1 covers higher-order cases (e.g. $\sin$); Prop. 1: squared loss on many base functions $g$ (tanh, sigmoid, exp, monomials, …) qualifies; Prop. 2: compositions too.
**Theorem 2.** $f(x)=\tfrac14(x^2-\mu)^2$, $\eta=K/\mu$, $1<K<\sqrt{4.5}-1\approx1.121$, $x_0\in(0,\sqrt\mu)$: GD converges to a period-2 orbit except for a measure-zero set of starts.
Theorem 4 / Prop. 4: a teacher–student single neuron converges to a period-2 orbit; Theorem 6 (App. J.1): the Theorem 1 condition holds at every minimum of matrix factorization.
### Notes' example
$\mu=1$: $f''(1)=2$, $f'''(1)=6$, $f''''(1)=6$, so $3\cdot36-2\cdot6=96>0$. With $\eta=1.05$, $x_0=0.8$: $x_1=1.1024$, $x_2\approx0.8532$.
> Check script: the 2-cycle exists for $K\in(1,1.237)$; period 4 at $K=1.25$, period 8 at $K=1.295$, then chaos. So Theorem 2's range is sufficient, not where the cycle ends.
