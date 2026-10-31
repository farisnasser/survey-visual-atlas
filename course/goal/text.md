## 1. A model is a function with knobs

A machine-learning model takes an input $x$ and produces a prediction $\hat y$ (read "y-hat"). What the prediction is depends on a set of adjustable numbers, the **[[parameters]]**, collected into one vector $\theta$ (the Greek letter theta).

The smallest useful example is a straight line:

$$\hat y = w\,x + b, \qquad \theta = (w, b).$$

Here $w$ is the slope and $b$ the intercept. Changing $w$ tilts the line; changing $b$ moves it up or down. A neural network is the same idea with many more knobs: every weight and every bias is one entry of $\theta$. We write $d$ for the number of entries, so $\theta \in \mathbb{R}^d$, a list of $d$ real numbers. For the line, $d = 2$. For a modern network, $d$ is in the millions or billions.

## 2. One number for "how wrong": the loss

We have $n$ training examples: pairs $(x_i, y_i)$ for $i = 1, \dots, n$, where $y_i$ is the correct answer for input $x_i$. To judge a choice of $\theta$, we compare each prediction with its answer and average the mistakes. That average is the **[[loss]]**:

$$\mathcal{L}(\theta) \;=\; \frac{1}{n} \sum_{i=1}^{n} \ell\big(\hat y_i, y_i\big).$$

The symbol $\sum_{i=1}^{n}$ means "add up the terms for $i = 1$ to $n$". The per-example loss $\ell$ measures one mistake. For predicting numbers, a common choice is the **squared error** $\ell(\hat y, y) = (\hat y - y)^2$. For the line this gives

$$\mathcal{L}(w, b) \;=\; \frac{1}{n} \sum_{i=1}^{n} \big(w\,x_i + b - y_i\big)^2 .$$

Other tasks use other per-example losses (classification typically uses the *cross-entropy*), but the idea is always the same: the loss turns "how good is this model?" into **one number that is small when the predictions are good**.

The loss depends on the data, but the data is fixed during training. The only thing we change is $\theta$. So we can think of $\mathcal{L}$ as a function of $\theta$ alone, and draw it. With two parameters, the loss is a height above the $(w, b)$ plane: a surface. This surface is called the **[[landscape|loss landscape]]**.

:::viz fit
:::

## 3. Training means minimizing

Training is the search for the parameters with the smallest loss:

$$\theta^\star \;=\; \operatorname*{arg\,min}_{\theta \in \mathbb{R}^d}\; \mathcal{L}(\theta).$$

[[argmin|$\operatorname{arg\,min}$]] means "the input that gives the smallest output" (not the smallest output itself, which is $\min \mathcal{L} = \mathcal{L}(\theta^\star)$). The star marks an optimal value.

In the visual above, the eight points give the best line $w^\star \approx 0.776$, $b^\star \approx 0.285$, with loss $\mathcal{L}^\star \approx 0.0086$. The flat line $w = b = 0$ has loss $\approx 1.03$; the line $w = 1, b = 0$ has loss $\approx 0.126$. Every choice of $(w, b)$ is a point on the landscape, and training means walking to its lowest point.

## 4. The landscape can have many valleys

The line-fitting landscape is a single bowl: one lowest point, and every path downhill leads to it. Neural-network landscapes are not like that. They can have many valleys, and we need words to tell them apart.

:::def Global and local minimum
A point $\theta^\star$ is a **[[global-min|global minimum]]** if no point is lower: $\mathcal{L}(\theta^\star) \le \mathcal{L}(\theta)$ for every $\theta$.

It is a **[[local-min|local minimum]]** if no *nearby* point is lower: there is a radius $r > 0$ such that $\mathcal{L}(\theta^\star) \le \mathcal{L}(\theta)$ whenever $\lVert \theta - \theta^\star \rVert < r$.
:::

Here $\lVert v \rVert = \sqrt{v_1^2 + v_2^2 + \dots + v_d^2}$ is the length of a vector $v$, its **[[norm]]**, so $\lVert \theta - \theta^\star \rVert$ is the distance between two parameter vectors. Every global minimum is also a local one; the reverse is not true.

The curve below is the one-parameter loss $\mathcal{L}(\theta) = 0.1\,\theta^2 + 0.5\sin(3\theta)$ on $-3 \le \theta \le 3$. It has three local minima, at $\theta \approx -2.50$, $-0.50$ and $1.50$, with losses $\approx 0.156$, $-0.474$ and $-0.264$. Only the middle one is global. A ball that simply rolls downhill ends in whichever valley it starts above.

:::viz valleys
:::

Two facts follow. First, *where you start* can decide *where you end*. Second, a method that only ever moves downhill can stop in a local minimum that is not the global one. Whether this actually hurts neural networks is one of the big questions of the field (stop 11).

## 5. Why we cannot just solve for the bottom

For the line, calculus gives the answer directly. At the bottom of the bowl, the loss is flat in both directions, which gives two linear equations in $w$ and $b$; solving them gives $w^\star$ and $b^\star$ exactly (this is the method of *least squares*, and stop 2 explains the "flat" condition).

For a neural network there is no such formula. The "flat" condition becomes millions of coupled nonlinear equations, and it is satisfied at many points that are not minima at all (stop 5). Finding the global minimum of a non-convex function is even provably hard: Murty and Kabadi showed that it is NP-hard already for quadratic objectives with simple constraints, and that even checking whether a given point is a local minimum is hard (stop 12).

So practical training is an **[[iterative|iterative method]]**: start at some $\theta_0$, then improve it step by step,

$$\theta_0 \;\to\; \theta_1 \;\to\; \theta_2 \;\to\; \cdots$$

with $\mathcal{L}$ going down along the way. Each step needs two decisions: **which direction** to move, and **how far**. The next two stops answer them.

## 6. Notation used from here on

Optimization papers write the same objects with different letters, and the rest of this course uses theirs, so the papers read naturally:

| ML notation | Optimization papers | Meaning |
|---|---|---|
| $\mathcal{L}(\theta)$ | $f(x)$ | the function we minimize (the loss) |
| $\theta \in \mathbb{R}^d$ | $x \in \mathbb{R}^d$ | the point we move (the parameters) |
| $\theta_t$ | $x_t$ | the point after $t$ steps |
| $\mathcal{L}^\star$ | $f^\star$ | the lowest value, $\inf_x f(x)$ |

Careful: in the papers, $x$ is the *parameter vector*, not a data input, and the letter $L$ (plain, not curly) is reserved for a smoothness constant (stop 3).

:::atlas Where this shows up in the Atlas
- [Murty & Kabadi 1987](paper.html?id=murty1987some) prove the hardness results of section 5.
- [Li et al. 2018](paper.html?id=li2018visualizing) draw neural-network landscapes as 2-D slices through the $d$-dimensional parameter space, the way the visuals here draw a 2-parameter loss.
- [Goodfellow et al. 2015](paper.html?id=goodfellow2015qualitatively) look at the loss along the straight line from the starting point $\theta_0$ to the trained $\theta$, a 1-D slice like the curve above.
:::

:::problem The next problem
We decided to improve $\theta$ step by step. But standing at one point of a landscape we cannot see, **which way is down?**
:::
