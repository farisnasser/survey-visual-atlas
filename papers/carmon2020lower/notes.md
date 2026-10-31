# Lower Bounds for Finding Stationary Points I

:::minute The paper in one minute
**Problem.** Optimization papers compete on rates such as $O(\epsilon^{-2})$ gradient calls or $O(\epsilon^{-3/2})$ Hessian-assisted iterations. Such numbers are upper bounds. Carmon et al. ask the matching question: *can any algorithm, even an extremely clever one, beat these dependences for all smooth non-convex objectives?*

**Key result.** The paper constructs smooth hard instances with Lipschitz $p$-th order derivatives and proves a family of oracle lower bounds, $\Omega\big(\epsilon^{-(p+1)/p}\big)$, suppressing smoothness and initial-gap constants. They match the classical gradient-descent rate for $p = 1$ and higher-order regularization rates in the appropriate oracle framework.

**Avoid conflation.** A lower bound for $\|\nabla f(x)\| \le \epsilon$ is not a lower bound for escaping all saddles or finding a global minimum. Those are stronger goals.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Upper bounds and lower bounds answer different questions

An **upper bound** comes from an algorithm: "this method always succeeds within $T$ queries." A **lower bound** is a statement about every algorithm: "for any method there is an instance that needs at least $T$ queries." When the two match, the method is **worst-case optimal**. A familiar example: guess a number between $1$ and $1024$ with yes/no questions. Halving the range each time needs $10$ questions (upper bound). Any strategy needs $10$ in the worst case, because each answer can at best halve the set of remaining candidates, and $2^{10} = 1024$ (lower bound). The two meet, so halving is optimal.

### 2. A chain function hides its coordinates one at a time

Take $f(x) = \tfrac12(x_1 - 1)^2 + \tfrac12\sum_{i=1}^{d-1}(x_{i+1} - x_i)^2$ and start at $x = 0$. The gradient there is nonzero only in coordinate $1$. After one gradient step only $x_1$ has moved; the gradient then touches coordinate $2$; and so on. With $d = 5$, after $k$ steps exactly the first $k$ coordinates are nonzero. A method that only learns about $f$ through derivatives at points it has visited discovers one new coordinate per query. The paper calls such functions **zero-chains** and builds its hard instances from them, with a construction that also keeps the gradient large until the whole chain has been uncovered.

### 3. Reading the exponents as query counts

The bounds are of the form $\epsilon^{-(p+1)/p}$. For a target $\epsilon = 0.01$: $p = 1$ gives $\epsilon^{-2} = 10{,}000$, $p = 2$ gives $\epsilon^{-3/2} = 1{,}000$, $p = 3$ gives $\epsilon^{-4/3} \approx 464$. Each extra order of controlled smoothness lowers the exponent, but the gains shrink, and higher-order methods pay more per query. These are counts of oracle calls up to constants, not of seconds.

:::key Putting the three ideas together
A lower bound limits every algorithm, not just one (idea 1). The paper's hard functions are chains that reveal one new direction per query while the gradient stays large (idea 2), which forces the query counts $\epsilon^{-(p+1)/p}$ (idea 3). Because they match the upper bounds of gradient descent and cubic-regularized Newton, those methods are worst-case optimal in their classes, as the notes below explain.
:::
:::

:::divider Study notes
:::

## 1. Problem: how fast could any algorithm possibly be?

Optimization papers compete on rates such as $O(\epsilon^{-2})$ gradient calls or $O(\epsilon^{-3/2})$ Hessian-assisted iterations. Such numbers are upper bounds. Carmon et al. ask the matching question: *can any algorithm, even an extremely clever one, beat these dependences for all smooth non-convex objectives?* They study the problem of finding $x$ with $\|\nabla f(x)\| \le \epsilon$, not global minimization and not second-order stationarity.

## 2. The oracle model and what the lower bound means

An algorithm queries a function at points $x_t$ and receives derivative information (in their powerful model, potentially derivatives through arbitrary finite order). The adversary can choose a smooth, high-dimensional function that is particularly difficult for that algorithm. The dimension is allowed to be sufficiently large for the worst-case construction.

$$\text{target:}\;\; \|\nabla f(x)\| \le \epsilon, \qquad \text{cost: number of oracle queries needed in worst case.}$$

A lower bound $\Omega(\epsilon^{-a})$ means that *some* admissible objective forces at least this many queries; it does not say that actual neural networks normally need that many iterations.

## 3. Key result and interpretation

The paper constructs smooth hard instances with Lipschitz $p$-th order derivatives and proves a family of oracle lower bounds, suppressing smoothness and initial-gap constants:

$$\boxed{\Omega\big(\epsilon^{-(p+1)/p}\big).}$$

The exponent depends on how many derivatives are Lipschitz-controlled:

| $p$ | smoothness class | exponent |
|:-:|:-:|:-:|
| 1 | Lipschitz gradient | 2 |
| 2 | Lipschitz Hessian | 3/2 |
| 3 | Lipschitz third derivative | 4/3 |

The paper's lower bounds are sharp to constants within their relevant classes; they match the classical gradient-descent rate for $p = 1$ and higher-order regularization rates in the appropriate oracle framework.

## 4. Why a path of hidden progress causes difficulty

**Intuition, not an exact description of the authors' construction.** Imagine a loss in which the gradient points along one informative coordinate at a time. Before progressing, an algorithm must uncover the next direction. A powerful local oracle cannot reveal arbitrarily distant stages of this progression. Smooth interpolation makes the construction differentiable while hiding its later stages from early queries. The actual proof uses carefully designed high-dimensional hard instances and information arguments; one should not substitute this paragraph for its formal lower-bound construction.

## 5. Match the result to gradient descent

For $L$-smooth $f$ bounded below by $f_{\inf}$, the standard descent lemma gives

$$f(x_{t+1}) \le f(x_t) - \frac{1}{2L}\|\nabla f(x_t)\|^2,$$

so summing yields

$$\min_{t<T}\|\nabla f(x_t)\|^2 \le \frac{2L(f(x_0) - f_{\inf})}{T}.$$

Thus gradient descent achieves $T = O(\epsilon^{-2})$ iterations for a first-order stationary point. The paper's $p = 1$ result shows this exponent is *worst-case optimal* in the corresponding function class. Adding Hessian smoothness changes the class and allows an exponent $3/2$ for higher-order methods, but usually adds expensive model subproblems.

:::theorem Reported contribution
Carmon et al. prove lower bounds for *first-order stationary points* under higher-order smoothness, including randomized algorithms in their stated oracle model. The result explains why many classical complexity exponents cannot generally be improved without strengthening assumptions or changing the information model.
:::

## 6. Limitations and survey relevance

Worst-case smooth functions need not resemble an SGD-trained, nonsmooth ReLU/Transformer loss. The oracle model neglects minibatching, parallel accelerators, memory, communication and tuning. It does not prove optimality of perturbed GD for second-order stationary points, and it does not rule out faster algorithms for restricted neural-network architectures. This paper supports **RQ3** (complexity frontiers) and **RQ4** (what the complexity model actually assumes).

:::warn Avoid conflation
A lower bound for $\|\nabla f(x)\| \le \epsilon$ is not a lower bound for escaping all saddles or finding a global minimum. Those are stronger goals.
:::

## 7. Our notes

This paper tells us how to interpret the many $1/\epsilon$ exponents in our complexity table. Gradient descent's $\epsilon^{-2}$ rate is not just a weak proof artifact for general smooth problems: there are worst-case functions that force it. But these are oracle lower bounds for first-order stationarity, not predictions of real neural-network training time.

> **Original text:** [https://doi.org/10.1007/s10107-019-01406-y](https://doi.org/10.1007/s10107-019-01406-y).
