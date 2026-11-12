## 1. The puzzle: one gradient, many decisions

This chapter follows Section I of the survey. Training a network means minimizing a training loss $L(\theta)$ over a long parameter vector $\theta$. Every optimizer in the survey starts from the same object, the **[[gradient]]** $g = \nabla L(\theta)$: the vector of slopes, which points uphill. The textbook rule is to step against it, $\theta \leftarrow \theta - \eta g$. But that rule leaves the survey's five questions open:

- **How accurate must the gradient be?** A full pass over the data, or a cheap minibatch estimate (Section III-A of the survey; chapter 5 here)?
- **Should the step remember the past?** Momentum adds previous gradients (chapter 7).
- **Is one step size enough?** Adam gives every coordinate its own (chapter 8).
- **Can curvature tell a steep narrow valley from a gentle plateau?** Newton's method uses the Hessian (chapters 10 to 12).
- **When a parameter is a matrix, should the update respect its rows and columns?** K-FAC, Shampoo and Muon do (chapters 13 to 16).

Each answer gives a different optimizer. They share an objective, but they differ in what information they collect, what notion of distance they use, and how much memory and arithmetic they cost.

## 2. The two research questions

The survey organizes everything around two questions.

:::key The survey's research questions
**RQ1. How do first-order optimization methods improve upon standard gradient descent?** Plain gradient descent uses only the current gradient and one learning rate, which is inefficient when gradients are noisy or when the loss varies very differently across parameter directions. RQ1 asks how first-order methods change the way gradient information is used, and what extra computation and memory those changes cost. Answered in Section III (chapters 7 to 9).

**RQ2. How do curvature-aware and matrix-aware methods extend gradient-based optimization?** Other methods build better updates from the curvature of the loss or from the structure of the parameters. RQ2 asks how they use that information and whether the improvement justifies the extra cost. Answered in Sections IV and V (chapters 10 to 16).
:::

The survey deliberately does not try to name a single best optimizer. Its aim is to understand why each approach was developed and under what conditions its advantages become meaningful. A method can make more progress per step but cost more per step. A theorem can hold for convex losses while a reported speedup comes from a transformer. Chapter 17 makes this precise.

## 3. A first look: the same problem, different updates

Here is the whole issue in one picture. Below, three optimizers start from the same point on the same loss and see the same gradients. They behave differently only because they transform those gradients differently.

:::viz race
:::

:::read How to read this visual
**What is drawn.** A three-dimensional view of a quadratic valley $f = \tfrac12 x^\top H x$ whose two curvatures are $0.1$ and $10$. Its condition number is $100$: one direction is a hundred times more curved than the other. Height is drawn as $\sqrt f$ so that the floor of the valley is visible. The green dot is the minimum.

**What moves.** Three balls start at $(-1.3, 1.2)$. Violet is gradient descent with its best fixed step; white is AdaGrad and coral is Adam (both give each coordinate its own step size, chapter 8). The chart below plots the loss of each on a log scale over 400 steps.

**What to try.** Run it, then turn the valley by $45^\circ$ with the rotation control and run it again.

**What to notice.** When the valley lines up with the axes, the per-coordinate methods race ahead: dividing each coordinate by its own gradient scale is exactly the right correction. Turned by $45^\circ$, the advantage disappears, because a per-coordinate scale cannot see a valley that runs diagonally. This one picture is the reason the survey separates *diagonal* methods (chapter 8) from *full-matrix* and *structured* methods (chapters 10 to 16). Chapter 4 explains it with the Hessian.
:::

## 4. The map of the survey

The survey's Fig. 1 lays out the progression. The first row is Section III: gradient descent and SGD, then momentum, then per-coordinate scaling. Below it, Sections IV and V split into four branches that look algebraically similar (each multiplies the gradient by some matrix) but estimate different things.

:::survey Fig. 1 of the survey: organization of the survey. The branches separate Hessian curvature, information geometry, gradient-statistical preconditioning, and matrix orthogonalization; their similar algebra does not make them the same approximation.
public/survey3/fig1.webp
:::

:::read How to read this figure
**What is drawn.** A flow chart. The top row (Section III, RQ1) runs from full-gradient GD and minibatch SGD to momentum and then to the per-coordinate methods AdaGrad, RMSProp, Adam, AMSGrad, AdamW, Adafactor and Lion. Each box names the survey equations that define its methods.

**The four branches (Sections IV and V, RQ2).** Each box ends with an italic line saying what that branch actually estimates:
- **Hessian curvature:** Newton, its safeguards, BFGS, Hessian-free methods, and the diagonal estimators. It estimates the Hessian $H$, its inverse, or its diagonal.
- **Information geometry:** the Fisher matrix, natural gradient and K-FAC. It estimates the Fisher $F$, not $H$.
- **Gradient-statistical preconditioning:** Shampoo and SOAP. They accumulate products of gradients, $GG^\top$ and $G^\top G$.
- **Matrix orthogonalization:** Muon. It normalizes singular values and estimates no curvature at all.

**What to notice.** The survey's main warning is visible in the branching. All four multiply the gradient by a matrix, so their formulas look alike, but "curvature" is not a synonym for every matrix-valued update. The rest of this guide follows the figure from top to bottom.
:::

## 5. How this guide works

Each chapter follows one part of the survey, in order, and keeps the survey's equation numbers. Words in dotted underline open a definition. Every live visual runs a real optimizer on a real function, and is followed by a box explaining what is drawn, what moves, and what to notice. The survey's own figures appear where the text discusses them. Its appendices are folded into the chapters they explain: the rotated valley of Appendices A and C in chapter 4, the SGD bound of Appendix C-B in chapter 5, Adam's bias correction of Appendix A-B in chapter 8, the saddle of Appendix D-A in chapter 10, the KL derivation of Appendix C-D in chapter 12, and the polar factor and Newton–Schulz iteration of Appendices A-C and C-C in chapter 16.

:::survey Fig. 10 of the survey: the analytic comparisons recommended in Section VI-F. (a) GD, heavy-ball momentum, Adam and exact Newton as balls rolling over one 3D landscape; (b, c) four updates built from one matrix gradient.
public/survey3/fig10.webp
:::

:::read How to read this figure
**Panel (a).** The curved valley $L = (1-x)^2 + 10(y - x^2)^2$ as a 3D surface (height $\log(1 + L)$), with its minimum $(1, 1)$ marked by the yellow star. All four methods start from the white ball at $(-1.2, 1)$. Gradient descent (blue, $\eta = 0.01$), heavy-ball momentum (orange) and Adam (green) follow the curved floor for 300 steps each; exact Newton (vermillion) reaches the minimum in 8 large steps, using the Hessian. Chapter 11 runs this race live.

**Panel (b).** One controlled $8\times8$ gradient matrix $G$ with singular values $4, 2, 1, \dots, 0.03$, and the singular values of the update each method builds from it, scaled so the largest is 1. SGD keeps the spectrum of $G$. Adam's first step divides each entry by its own size, which does not flatten the spectrum. One Shampoo step and Muon both give all singular values equal to one.

**Panel (c).** The four update matrices as colour maps. Shampoo and Muon coincide here because Shampoo has no history yet; chapter 16 explains why. The picture is a toy: it shows what each update does to a matrix, not which one trains networks best.
:::

:::problem The next step
Before any optimizer, the survey builds its tools: the Taylor models that every method is secretly minimizing. Chapter 2 starts there.
:::
