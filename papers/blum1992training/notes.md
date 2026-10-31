# Training a Tiny Network Can Be NP-Complete

:::minute The paper in one minute
**Problem:** Does making a neural network extremely small make finding exactly consistent weights easy?

**Method:** A polynomial-time reduction from a combinatorial decision problem to whether a particular three-node threshold network fits all labeled examples.

**Result:** This training-consistency decision problem is NP-complete.

**Why we care:** Neural-network training can be intrinsically difficult in the worst case, even with very few computing units.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. One threshold unit draws one flat cut

A **threshold unit** outputs $1$ when $w^\top x \ge b$ and $0$ otherwise. The boundary $w^\top x = b$ is a hyperplane, so the unit can only say which side of one flat cut an input is on. On the four corners of the square $\{0,1\}^2$, AND is easy: $\mathbf{1}\{x_1 + x_2 \ge 2\}$. XOR is impossible. It would need $w_1 \ge b$ and $w_2 \ge b$ (inputs $10$ and $01$ are positive), $0 < b$ (input $00$ is negative) and $w_1 + w_2 < b$ (input $11$ is negative). The first three give $w_1 + w_2 \ge 2b > b$, which contradicts the fourth.

### 2. Two hidden units buy more than one cut

Add a hidden layer of two threshold units and feed their outputs to a third. With $h_1 = \mathbf{1}\{x_1 + x_2 \ge 1\}$, $h_2 = \mathbf{1}\{x_1 + x_2 \ge 2\}$ and output $\hat y = \mathbf{1}\{h_1 - h_2 \ge 1\}$, the network outputs $0, 1, 1, 0$ on $00, 10, 01, 11$: it computes XOR. The price is a new decision. For each training example, the learner must also choose what the hidden units say about it.

### 3. NP-complete means "easy to check, believed hard to find"

A decision problem is in **NP** when a "yes" answer has a certificate that can be checked in polynomial time. Here the certificate is the set of weights: plug them in and test all $m$ examples. The paper notes, citing Raghavan, that $O(n \log n)$ bits per weight suffice, so the certificate is short. A problem is **NP-complete** when it is in NP and every NP problem reduces to it. No polynomial-time algorithm is known for any NP-complete problem.

### 4. The reduction turns a set-splitting puzzle into points on a cube

The paper starts from **Set-Splitting**: split a finite set $S$ into $S_1, S_2$ so that no given subset $c_j$ lies wholly inside one part. Each element $s_i$ becomes a negative point at the unit vector $p_i$, the origin is positive, and each subset $c_j$ becomes a positive point at the sum of its unit vectors. The paper's own example is $S = \{s_1, s_2, s_3\}$, $c_1 = \{s_1, s_2\}$, $c_2 = \{s_2, s_3\}$: negatives at $100, 010, 001$, positives at $000, 110, 011$. The split $S_1 = \{s_1, s_3\}$, $S_2 = \{s_2\}$ gives the paper's two planes $a^\top x = -\tfrac12$ and $b^\top x = -\tfrac12$ with $a = (-1, 3, -1)$ and $b = (3, -1, 3)$ (weight $-1$ inside the part, $n = 3$ outside). All three positives have $a^\top x \ge 0$ and $b^\top x \ge 0$, while each negative falls below $-\tfrac12$ on one of the two planes.

:::key Putting the four ideas together
A single unit is a linear program (idea 1), but the hidden layer adds a discrete choice per example (idea 2). Checking weights is easy (idea 3), yet the reduction plants Set-Splitting inside the choice of two planes (idea 4), so deciding whether consistent weights exist is NP-complete. The notes below state exactly which network and which question this covers.
:::
:::

:::divider Study notes
:::

## 1. Define the exact network before discussing hardness

Here "three nodes" means a *two-layer threshold network* with two hidden threshold units and one output threshold unit (not a modern three-layer ReLU MLP). A threshold neuron computes

$$h(x) = \mathbf{1}\{w^\top x \ge b\}, \quad x \in \{0,1\}^n \text{ or another specified input domain.}$$

The hidden layer produces $h_1(x), h_2(x) \in \{0, 1\}$, and the output unit computes

$$\hat y(x) = \mathbf{1}\{a_1 h_1(x) + a_2 h_2(x) \ge c\}.$$

**A-level reminder.** The inequality $w^\top x \ge b$ defines a half-space. Each hidden neuron decides which side of a hyperplane an input lies on. Training chooses the hyperplanes and output rule jointly.

## 2. The precise decision problem

Given a finite labeled training set $S = \{(x_i, y_i)\}_{i=1}^m$, ask whether there exist thresholds and weights such that

$$\forall i \in \{1, \dots, m\}, \qquad \hat y(x_i) = y_i.$$

This is an *existence-of-perfect-fit* decision problem. Equivalently, for the zero–one empirical error

$$R(\theta) = \sum_{i=1}^m \mathbf{1}\{\hat y_\theta(x_i) \ne y_i\},$$

we ask whether $\min_\theta R(\theta) = 0$. The equation here restates the decision problem; it is not a differentiable loss suitable for ordinary backpropagation.

## 3. Why a single perceptron is different

For a *single* linear threshold unit, finding a consistent separator can be formulated as linear inequalities, e.g. for labels $s_i \in \{-1, +1\}$,

$$s_i(w^\top x_i - b) \ge 1 \qquad \text{for every } i,$$

when a strictly separating solution exists and can be rescaled. Linear-programming feasibility is polynomial-time solvable. Adding just a small hidden layer changes the logical structure: the learner must decide *which hidden-unit regions* should account for each sample. The combination creates a combinatorial search problem.

## 4. What the hardness proof establishes

The authors encode combinatorial consistency constraints into examples for their specific threshold architecture and prove NP-completeness of perfect fitting. They also discuss other simple networks and the dependence of difficulty on representation. The proof does *not* say that arbitrary SGD runs on differentiable ReLU/tanh networks are NP-complete, nor that typical training examples exhibit the constructed worst-case difficulty.

:::warn Scope matters.
The paper itself explicitly leaves extension to nonlinear units such as sigmoids as an open question. It proves hardness for *threshold* neurons and exact training consistency, not for every activation, loss, or approximate optimization target.
:::

## 5. Why the result matters to our survey

**RQ4:** Mathematical guarantees about benign optimization must be read with architectural, data and loss assumptions attached. Worst-case hardness of a tiny threshold network can coexist with efficient convergence theorems for highly overparameterized, specially initialized smooth or ReLU networks.

:::key Our takeaway.
A tiny neural network is not automatically easy to train. This is a worst-case existence result, so it motivates the rest of our survey rather than contradicting empirical success of modern optimization.
:::

> **Sources:** Journal DOI [https://doi.org/10.1016/S0893-6080(05)80010-3](https://doi.org/10.1016/S0893-6080%2805%2980010-3); author-hosted journal reprint [https://people.csail.mit.edu/rivest/pubs/BR93.pdf](https://people.csail.mit.edu/rivest/pubs/BR93.pdf).
