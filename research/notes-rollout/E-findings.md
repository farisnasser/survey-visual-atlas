# Session E — in-page notes findings

## Shared changes needed
None so far.

## [1] zhang2021escape — Escape Saddle Points by a Simple Gradient-Descent Based Algorithm
- **Checked against:** NeurIPS 2021 proceedings PDF (the copy the notes link), pp. 1–10: Eq. (1), Table 1, Algorithm 1 + Proposition 3 (p. 5), Eq. (5)–(6), Remark 4, Algorithm 2 + Proposition 5 (p. 7), Lemma 6 + Theorem 7 (p. 8), Algorithm 3 + Theorem 9 (p. 9), Sec. 4 and Figures 1–2 (p. 10).
- **Fixes to the notes:** none. No discrepancies found.
- **Observations (no change made):**
  - The notes bound the gradient-difference error by $\tfrac{\rho}{2}\|u\|^2$; the paper's Eq. (6) states the looser $\rho\|x\|^2$. Both are true (integrating $\|H(\xi x) - H(0)\| \le \rho\xi\|x\|$ over $\xi$ gives $\rho/2$); the check script verifies the $\rho/2$ bound numerically.
  - The notes write the dimension as $d$; the paper uses $n$. Notation only.
  - "NEON/NEON2 (Refs 76–77)" uses the notes' old reference numbering (survey numbers: Neon = [81], Neon2 = [33]). Kept verbatim; the coordinator may want to decide one policy for old "Ref" numbers across all pages.
  - The notes have no "Primary source" line; their closing "Verified source: original NeurIPS 2021 PDF, …" line is kept verbatim with its link.
  - The two light-blue boxes without titles ("Important:", "Read:") are kept as untitled `:::warn` / `:::plain` boxes with the bold label inline, as in the PDF.
- **Unverified:** nothing.
- **Primer ideas:** (1) second-order stationarity and why negative curvature guarantees a decrease $\gamma^3/(3\rho^2)$; (2) a difference of two gradients is a Hessian-vector product with error $\le \rho/2\|u\|^2$ (worked example); (3) GD near a saddle is a power method, and a random direction's $1/\sqrt d$ overlap explains the $\log d$ (694 vs 1388 steps for $d = 10^6, 10^{12}$); (4) counting escapes (Lemma 6 decrease) times the per-escape cost (Proposition 5) gives $\varepsilon^{-7/4}$.

## [2] wang2019spiderboost — SpiderBoost and Momentum
- **Checked against:** NeurIPS 2019 proceedings PDF (the copy the notes link), pp. 1–8: Eq. (1) (p. 3), Algorithms 1–2 (p. 4), Assumption 1 + Theorem 1 (p. 4), Sec. 3 Fact 1 + Theorem 2 (p. 5), Sec. 4 Algorithm 3 + Theorem 3 (pp. 6–7), Sec. 5 experiments (Figures 1–3); arXiv 1810.10690 has the same section layout.
- **Fix to the notes (1):**
  - Reading priority box. Original: "Section 2 (Eq. 1, Algorithms 1–2 and Theorems 1–2), Section 3 (Algorithm 3, Theorem 3), and Figures in Section 5." Fixed: "Sections 2–3 (Eq. 1, Algorithms 1–2 and Theorems 1–2), Section 4 (Algorithm 3, Theorem 3), and Figures in Section 5." Paper: Sec. 2 "SpiderBoost for Nonconvex Optimization" holds Eq. (1) and Theorem 1; Sec. 3 "Prox-SpiderBoost for Nonconvex Composite Optimization" holds Theorem 2 (Algorithm 2 is printed on p. 4 but introduced in Sec. 3.2); Sec. 4 "Accelerating Prox-SpiderBoost via Momentum" holds Algorithm 3 and Theorem 3 (p. 6–7).
- **Observations (no change made):** the notes' Eq. (1) is written as $v_{k-1} + \frac{1}{|S_k|}\sum(\cdot)$; the paper writes $\frac{1}{|S|}\sum(\nabla f_i(x_k) - \nabla f_i(x_{k-1}) + v_{k-1})$ — identical. "takes two coupled proximal updates" is loose (the paper's $y_{k+1}$ reuses the same prox term as $x_{k+1}$, so only one prox is computed; Sec. 4.1 says so) but not wrong. Old reference numbers "SSRGD (Ref 83)", "Stabilized SVRG (Ref 82)", "NEON2 (Ref 77)" kept verbatim (survey: [87], [86], [33]).
- **Unverified:** nothing.
- **Primer ideas:** (1) SFO counting: $n\varepsilon^{-2} = 10^{10}$ vs $n + \sqrt n\varepsilon^{-2} = 1.1\times10^7$; (2) single-sample gradients keep variance 1 at the minimizer of the notes' pair; (3) the recursive estimator's error grows with movement, resets every $\sqrt n$ steps cost about $3\sqrt n$ per iteration; (4) proximal step and gradient mapping, $G_1(0.7) = 0$, $G_1(0) = -0.7$ for $\tfrac12(x-1)^2 + 0.3|x|$.

## [3] allenzhu2018natasha — Natasha 2: Faster Non-Convex Optimization Than SGD
- **Checked against:** NeurIPS 2018 proceedings PDF (the copy the notes link): Sec. 1.2 "Our Resolutions" with Figure 2 (a: negative curvature via Oja, b: safe zone) and informal Theorems 1–2 (p. 3), Sec. 3 assumptions (A1)–(A3) and Theorem 1, Sec. 4 (A4), Definition 4.2, Theorem 2 (informal, $\delta \in (0, \varepsilon^{1/4})$), Remarks 4.4–4.5, Corollaries 4.6–4.7, Algorithm 2 (pp. 7–9).
- **Fixes to the notes:** none. No discrepancies found.
- **Observations (no change made):** the notes' $\widetilde O(\delta^{-5} + \delta^{-1}\varepsilon^{-3} + \varepsilon^{-13/4})$ and "taking $\delta \ge \varepsilon^{1/4}$ yields $\widetilde O(\varepsilon^{-13/4})$" are exactly the intro's Theorem 2 and Remark 4.4 (the Sec. 4 statement itself needs $\delta < \varepsilon^{1/4}$; Remark 4.4 extends it by replacing $\delta$ with $\varepsilon^{1/4}$). The surrogate $f + \sigma\|x - \hat x\|^2$ is "representative": Natasha 2 itself applies Natasha 1.5 to $F^k = f + L(\max\{0, \|x - y_k\| - \delta/L_2\})^2$ (Algorithm 2, line 9), and the final step uses $+\tilde\sigma\|x - \hat y\|^2$ (line 17); the notes say "representative", so kept. Old reference numbers (Refs 77, 80, 83) kept verbatim.
- **Unverified:** nothing.
- **Primer ideas:** (1) online cost counted in stochastic gradients + Hessian-vector products, $\varepsilon^{-4} = 10^{16}$ vs $\varepsilon^{-3.25} = 10^{13}$; (2) Oja = power method with one sampled Hessian per step, $(1.1/0.9)^{30} \approx 412$; (3) $\sigma$-nonconvexity cancelled by $\sigma\|x - \hat x\|^2$ (the notes' function is 1-nonconvex); (4) Hessian smoothness gives a safe zone of radius $\delta/L_2$ and $\delta = \varepsilon^{1/4}$ balances the two costs.
