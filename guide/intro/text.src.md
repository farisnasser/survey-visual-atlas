## 1. The puzzle: one gradient, many decisions

This chapter follows Section I of the survey. Training a network means minimizing a loss $L(\theta)$ by repeatedly updating its parameters $\theta$. Every optimizer in the survey starts from the same object, the **[[gradient]]** $g = \nabla L(\theta)$, which points uphill. The textbook rule is to step against it, $\theta \leftarrow \theta - \eta g$. But the gradient says how the loss changes; the optimizer still decides how to use that information:

- **How accurate must the gradient be?** A full pass over the data, or a cheap minibatch estimate (chapter 5)?
- **Should the step remember the past?** Momentum adds previous gradients (chapter 7).
- **Is one step size enough?** Adam gives every coordinate its own (chapter 8).
- **Can curvature tell a steep, narrow valley from a gentle plateau?** Newton's method uses the Hessian (chapters 10 to 12).
- **When a parameter is a matrix, should the update respect its rows and columns?** K-FAC, Shampoo and Muon do (chapters 13 to 16).

Each answer gives a different optimizer. They share an objective, but differ in what information they collect, what notion of distance they use, and how much memory and arithmetic they cost. Deep networks make every one of these questions matter: their losses are non-convex, curvature varies enormously across directions, full gradients are expensive, and optimizer state grows with the model.

## 2. The two research questions

Existing work splits along two lines: classical optimization theory emphasizes convergence guarantees, while research on modern optimizers often focuses on training efficiency. The survey connects them by asking how each method builds its update, what information it uses, and how its mathematics relates to its cost, for single-objective, unconstrained training by empirical risk minimization. It organizes everything around two questions.

:::key The survey's research questions
**RQ1. How do first-order optimization methods improve upon standard gradient descent?** Plain gradient descent uses only the current gradient and one learning rate, which is inefficient when gradients are noisy or when the loss varies very differently across parameter directions. RQ1 asks how first-order methods change the way gradient information is used, and what extra computation and memory those changes cost. Answered in Section III (chapters 7 to 9).

**RQ2. How do curvature-aware and matrix-aware methods extend gradient-based optimization?** Other methods build better updates from the curvature of the loss or from the structure of the parameters. RQ2 asks how they use that information and whether the improvement justifies the extra cost. Answered in Sections IV and V (chapters 10 to 16).
:::

The survey deliberately does not try to name a single best optimizer. Its aim is to understand why each approach was developed and under what conditions its advantages become meaningful. A method can make more progress per step but cost more per step; a theorem can hold for convex losses while a reported speedup comes from a transformer. Chapter 17 makes this precise.

## 3. A first look: the same problem, different updates

Here is the whole issue in one picture. Three optimizers start from the same point on the same loss and see the same gradients. They behave differently only because they transform those gradients differently.

@@viz race@@

## 4. The map of the survey

The survey's Fig. 1 lays out the progression. The first row is Section III: gradient descent and SGD, then momentum, then per-coordinate scaling. Below it, Sections IV and V split into four branches that look algebraically similar (each multiplies the gradient by some matrix) but use very different information.

:::survey Fig. 1 of the survey: organization of the survey. The branches separate Hessian curvature, information geometry, gradient-statistical preconditioning, and matrix orthogonalization; their similar algebra does not make them the same approximation.
public/survey3/fig1.webp
:::

:::read How to read this figure
**What is drawn.** A flow chart. The top row (Section III, RQ1) runs from GD and SGD, which step against the full or minibatch gradient, to momentum methods, which remember past directions, and then to the per-coordinate methods AdaGrad, RMSProp, Adam, AMSGrad, AdamW, Adafactor and Lion, which give every coordinate its own step size from gradient statistics.

**The four branches (Sections IV and V, RQ2).** Each box ends with an italic line saying what that branch uses:
- **Hessian curvature:** Newton and its safeguards, BFGS and L-BFGS, Hessian-free methods, and the diagonal estimators AdaHessian, Apollo and Sophia. They use how the gradient changes.
- **Information geometry:** the Fisher matrix, natural gradient and K-FAC. They use how the predictions change.
- **Gradient-statistical preconditioning:** Shampoo and SOAP. They use accumulated products of gradients, $GG^\top$ and $G^\top G$.
- **Matrix orthogonalization:** Muon. It equalizes the singular values of the update and uses no Hessian or Fisher at all.

**What to notice.** All four branches multiply the gradient by a matrix, so their formulas look alike, but "curvature" is not a synonym for every matrix-valued update. The rest of this guide follows the figure from top to bottom.
:::

## 5. How this guide works

Each chapter follows one part of the survey, in order, and quotes the survey's own equation, figure and table numbers. The explanations follow one pattern: start with a small example in actual numbers, define every symbol, derive one algebraic step at a time, say what it means, and connect it to optimization. Boxes marked *In plain English* give the intuition, and *Remember* boxes collect what to keep. Words in dotted underline open a definition. Every live visual runs a real optimizer on a real function and is followed by a box explaining what is drawn, what moves, and what to notice. The survey's own figures appear where the text discusses them.

The chapters are written from our [study notes for Sections II–VIII (PDF, 79 pages)](public/survey3/Study_Notes.pdf), which work through the same examples at greater length. Every chapter links to the page of the notes where its topic starts, under its title and in the top bar.

The worked numbers are teaching illustrations, chosen small enough to check by hand; they are not experiments. The only measured results are the survey's dashboard runs in chapter 17.

:::problem The next step
Before any optimizer, the survey builds its tools: what is being minimized, and the Taylor models that every method is secretly minimizing. Chapter 2 starts there.
:::
