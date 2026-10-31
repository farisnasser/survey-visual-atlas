# Crash-course rollout brief (sessions A, B, C): stops 4–12

Stops 1–3 are written and approved by Faris ("Looks good, continue"). Three sessions write the remaining nine stops in
parallel, three each, on their own branches; the coordinator (Faris's main chat) merges, audits and pushes.

## Read first
- **`.claude/skills/crash-course/SKILL.md` — the spec; follow it exactly.**
- Stops 1–3 in full: `course/goal|slope|steps/text.md`, their `viz.js`, `scripts/check-math/course_goal|slope|steps.py`.
  Open them in the browser (`learn.html?ch=goal` etc.) to see the bar for writing and for visuals.
- `research/course-plan.md` (the 12 stops), `course/course.json`, `course/glossary.json` (entries for your stops already
  exist; their `sec` anchors are still empty).
- `research/findings.md`: the verified statement of every paper you cite (theorem numbers, rates, assumptions, what the
  survey or notes got wrong). Look up each paper there before you mention it; read the paper itself if findings do not
  cover the claim.
- Existing paper visuals for ideas and verified toy models (e.g. `papers/lee2016gradient`, `papers/jin2017escape`,
  `papers/cohen2021gradient`, `papers/zhang2020gradient`, `papers/gupta2018shampoo`, `papers/carmon2020lower`): you may reuse
  their functions (and the numbers their check scripts verified), never their claims without checking.

## The story the three sessions share
Stop 3 ended with "one number $L$ hides how differently the landscape bends in different directions, and that difference is
what makes GD zig-zag". Each stop opens on the previous stop's `:::problem` and ends with its own. Keep the cross-references
("stop 7", "stop 9") consistent with this table:

| # | id | Title (keep) | Opens with | Must cover (minimum) | Ends with the problem |
|---|---|---|---|---|---|
| 4 | curvature | Curvature: how the landscape bends | zig-zag from stop 3 | second derivative → Hessian (symmetric), eigenvalues/eigenvectors as principal curvatures, quadratic model, GD decouples along eigen-directions ($u_i \leftarrow (1-\eta\lambda_i)u_i$), condition number and GD's rate on a quadratic, convex / strongly convex / PSD, Hessian-vector products and their cost | a zero gradient can still hide downward curvature: which flat points are traps? |
| 5 | critical | A zoo of flat points | the problem above | critical points (min, max, saddle, monkey saddle, degenerate), index, strict saddle, why saddles abound in high dimensions (Dauphin et al.'s argument, stated as they state it), saddles from symmetry ($(1-uv)^2$), FOSP vs SOSP, $\varepsilon$-SOSP with $\rho$ | GD can pass near saddles slowly: is it ever stuck, and how do we get out fast? |
| 6 | escape | Escaping saddle points | the problem above | escape along negative curvature ($(1+\eta\lvert\lambda\rvert)^t$, time $\sim \log(1/\text{offset})$), stable set + measure zero (Lee 2016 Thm 4), it can still take exponentially long (Du 2017), perturbed GD (Jin 2017), finding negative curvature with gradients (power iteration, NEON/NEON2), Newton is attracted to saddles, saddle-free Newton (Dauphin), cubic regularization (Nesterov–Polyak), a table of SOSP rates from findings | escaping is solved, yet GD is still slow in narrow valleys: can memory of past steps help? |
| 7 | memory | Giving the optimizer memory | zig-zag again (stops 3–4) | momentum / heavy ball (Polyak 1964), the EMA view, Nesterov's look-ahead, acceleration on quadratics ($\sqrt\kappa$ vs $\kappa$, derived from the 2×2 iteration matrix), momentum's overshoot, accelerated non-convex methods (Jin 2018), Lookahead (Zhang 2019) | full gradients over millions of examples are too expensive |
| 8 | noise | Noise: stochastic gradients | the problem above | finite-sum loss, minibatch gradient (unbiased, variance $\sigma^2/B$), SGD, constant step → noise floor, decaying steps and Robbins–Monro, epochs, $O(\varepsilon^{-4})$ (Ghadimi–Lan), variance reduction (SVRG / SPIDER / SpiderBoost, $\varepsilon^{-3}$), noise helping escape and its limits (CNC, heavy tails) | one learning rate for all coordinates is still wrong when scales differ |
| 9 | adaptive | A step size for every direction | the problem above | preconditioning (Newton as the ideal), AdaGrad, RMSProp, Adam (with bias correction), what is known about Adam's convergence (from findings: Li 2023, ADOPT), gradient clipping and $(L_0, L_1)$-smoothness, Shampoo, second-order-ish methods in the Atlas (SASSHA etc., only as findings support) | among many minima, which does training find, and does it matter? |
| 10 | sharpness | Sharp and flat minima | the problem above | sharpness $\lambda_{\max}$, stability $\eta < 2/\lambda_{\max}$ at a minimum, large steps reject sharp minima, edge of stability and progressive sharpening (Cohen 2021), self-stabilization (Damian 2023), flat vs sharp and generalization (Hochreiter–Schmidhuber 1997, Keskar 2017, Dinh 2017's reparameterization caveat), SAM (Foret 2021; Agarwala 2023), warmup (Kalra 2024) | non-convex, yet training works: what is special about network landscapes? |
| 11 | landscape | Why big networks are easier to train | the problem above | symmetry and continuous families of minima, no spurious minima results (deep linear: Baldi–Hornik, Laurent–Brecht; matrix problems: Ge 2017, Bhojanapalli 2016), spurious minima do exist (Safran–Shamir 2018), wide layers (Nguyen–Hein 2017), the PL condition, overparameterization and GD to zero loss (Du 2019, Allen-Zhu 2019), NTK / lazy training (Arora 2019, Chizat 2019), mean field (Chizat–Bach 2018, Mei 2019), residual connections (He 2016), what landscape plots show (Li 2018, Goodfellow 2015) | what is the best any algorithm could possibly do? |
| 12 | limits | How hard can it be? | the problem above | oracles, upper vs lower bounds, $\tilde O$, the zero-chain idea behind Carmon et al.'s lower bounds, $\Omega(\varepsilon^{-2})$ and $\varepsilon^{-(p+1)/p}$, stochastic lower bounds (Arjevani 2023, 2020), NP-hardness (Murty–Kabadi), a "complexity ladder" visual of every rate in the course with its paper, and a closing section: how to read a paper in the Atlas with what you now know (where to go next) | (none: point back to the Atlas map) |

## Sessions
- **A**: stops 4, 5, 6 (curvature, critical, escape) — port 8577.
- **B**: stops 7, 8, 9 (memory, noise, adaptive) — port 8578.
- **C**: stops 10, 11, 12 (sharpness, landscape, limits) — port 8579.

## Per stop
1. Plan sections and 3–5 visuals (at least one 3D); check every planned claim against findings.
2. Write the check script first (functions, gradients/Hessians by finite differences, every number), then `viz.js`, then
   `text.md`.
3. Glossary: fill the `sec` anchors of your stop's terms, refine definitions only if needed, add new terms (keep them in
   your stop's block of `course/glossary.json`), keep `course.json` `terms` in sync, set `"ready": true` when done.
4. Audit (SKILL.md) at 1280 px and 375 px, screenshots of every visual.
5. Append to `research/course-rollout/<SESSION>-notes.md`: per stop, the sections, the visuals (function, what it shows), every
   paper claim with its findings line or page, and anything you could not verify.

## Do not touch
`src/`, `learn.html`, `index.html`, `paper.html`, `papers/`, `data/`, other stops' folders and check scripts,
`scripts/check-math/common.py`, `scripts/check-math/results.txt`, `research/findings.md`, `CLAUDE.md`, `.claude/`.
In `course/course.json` and `course/glossary.json` change only your own stops' lines/entries (other sessions edit theirs; git
merges disjoint lines). If a shared change is needed (a renderer feature, a library helper), work around it and describe it
under "Shared changes needed" at the top of your notes file.

## Commit, do not push
Commit to your branch after each finished stop (message `Crash course <SESSION>: stop <n> <id>`, ending with the line
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Never push, merge or switch branches. Report at the end: stops
done, visuals per stop, every paper claim you could not verify, shared changes needed.
