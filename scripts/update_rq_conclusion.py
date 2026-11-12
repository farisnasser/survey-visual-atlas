"""One-off (2026-11-12): bring the guide's research questions (chapter 1) and the survey's answers (chapter 18) in line
with the final wording of survey v3 (Section I-B and Section VIII)."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "guide")

def rep(path, old_start, old_end, new):
    p = os.path.join(G, path); s = open(p, encoding="utf-8").read()
    a = s.index(old_start); b = s.index(old_end, a) + len(old_end)
    s = s[:a] + new + s[b:]; open(p, "w", encoding="utf-8").write(s); print("updated", path)

rep("intro/text.md", ":::key The survey's research questions", "Chapter 17 makes this precise.",
""":::key The survey's research questions
**RQ1. How do first-order optimization methods improve upon standard gradient descent?** Plain gradient descent uses only the current gradient and one learning rate, which is inefficient when gradients are noisy or when the loss varies very differently across parameter directions. RQ1 asks how first-order methods change the way gradient information is used, and what extra computation and memory those changes cost. Answered in Section III (chapters 7 to 9).

**RQ2. How do curvature-aware and matrix-aware methods extend gradient-based optimization?** Other methods build better updates from the curvature of the loss or from the structure of the parameters. RQ2 asks how they use that information and whether the improvement justifies the extra cost. Answered in Sections IV and V (chapters 10 to 16).
:::

The survey deliberately does not try to name a single best optimizer. Its aim is to understand why each approach was developed and under what conditions its advantages become meaningful. A method can make more progress per step but cost more per step. A theorem can hold for convex losses while a reported speedup comes from a transformer. Chapter 17 makes this precise.""")

rep("open/text.md", "## 7. Conclusion: the survey's answers", "without treating the newest algorithm as automatically the best.\n:::",
"""## 7. Conclusion: the survey's answers

This section follows Section VIII of the survey. Every method in it changes the way gradients are turned into parameter updates:
- **GD** follows the gradient in Euclidean geometry, and **SGD** estimates it from a minibatch.
- **Momentum** accumulates its history, and **Nesterov** evaluates it at a look-ahead point.
- **AdaGrad, RMSProp, Adam and AdamW** scale each coordinate by its gradient history; **Adafactor and Lion** do it with less memory.
- **Newton and quasi-Newton methods** add the curvature of the loss.
- **Natural gradient and K-FAC** use the sensitivity of the model's predictions, the Fisher geometry.
- **Shampoo and SOAP** use structured matrix statistics of the gradients.
- **AdaHessian, Apollo and Sophia** estimate diagonal curvature.
- **Muon** reshapes a momentum update along its singular directions.

:::key The survey's answers
**RQ1.** First-order methods extend gradient descent without estimating curvature. Momentum makes better use of previous gradient directions, adaptive methods scale each parameter by its gradient history, and the newer designs cut memory and simplify the update (chapters 7 to 9).

**RQ2.** Second-order methods use curvature to account for how the gradient changes; natural-gradient methods use the sensitivity of the predicted distribution; structured and matrix-aware methods exploit how the parameters are organized to make otherwise expensive updates affordable. These methods do not all estimate the same quantity, even when their update rules involve similar matrix operations (chapters 10 to 16).

**The overall lesson.** More informative updates do not automatically mean more efficient training: the gain has to be weighed against compute and memory, and a convergence guarantee is only as relevant as its assumptions. Understanding an optimizer means asking what information it uses and how that information shapes the update. That gives a way to choose methods by the training problem and the resources available, rather than a ranking with a universal winner.
:::""")
