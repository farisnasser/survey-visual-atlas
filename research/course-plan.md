# Crash course: optimization from first principles (plan)

Faris (2026-10-31): a part of the Atlas that "introduces all the terms in optimization, with the proper visuals, and
basically everything that you need to be able to digest the papers in the Atlas", told "like a journey" (we want to
minimize the loss; then this problem appears, so people came up with that fix), math "built from first principles",
"every weird new term or variable defined and explained in simple words", "the ultimate guide for anyone wanting to
understand optimization and learn what the field is like".

## Shape
- `learn.html` = the journey map (12 stops) · `learn.html?ch=<id>` = one chapter · `learn.html?ch=glossary` = all terms.
- Each chapter: a question (the problem that forces the next idea) → the math built up step by step → live visuals
  embedded between the paragraphs (`:::viz name`) → "where this shows up in the Atlas" (links to paper pages) → the
  next problem, which opens the next chapter.
- Every term is defined once in `course/glossary.json` (plain words + the formula + the chapter that builds it) and
  linked from the text as `[[id]]` / `[[id|shown text]]`; hovering or tapping shows the definition.
- Visuals follow the Atlas contract: real functions, real optimizers, every number shown is recomputed in
  `scripts/check-math/course_<id>.py`.

## The 12 stops
| # | id | Title | The problem it answers | Main terms |
|---|---|---|---|---|
| 1 | goal | The goal: make the loss small | What does "training" mean? | parameters θ, loss, landscape, argmin, global/local minimum |
| 2 | slope | Which way is down? | How do we know where to move? | derivative, partial derivative, gradient, contour, directional derivative, Taylor, Jacobian, chain rule |
| 3 | steps | How far to step | Too small = slow, too big = blow-up | gradient descent, step size η, Lipschitz, L-smooth, descent lemma, ε-stationary, iteration complexity, O(·) |
| 4 | curvature | Curvature | Why does GD zig-zag in valleys? | second derivative, Hessian, eigenvalues, PSD, convex, strongly convex, condition number |
| 5 | critical | A zoo of flat points | ∇L = 0 does not mean "done" | critical point, saddle, strict saddle, degenerate, FOSP, SOSP, Hessian-Lipschitz ρ |
| 6 | escape | Escaping saddles | GD can crawl past saddles | stable set, measure zero, perturbed GD, negative curvature, Newton, cubic regularization |
| 7 | memory | Giving the optimizer memory | Zig-zag and slow progress | momentum, heavy ball, Nesterov look-ahead, acceleration, Lookahead |
| 8 | noise | Noise | Full gradients are too expensive | SGD, minibatch, variance, Robbins–Monro, noise floor, variance reduction |
| 9 | adaptive | A step size per direction | One η cannot fit every direction | preconditioning, AdaGrad, RMSProp, Adam, bias correction, clipping, (L0,L1)-smooth, Shampoo |
| 10 | sharpness | Sharp and flat minima | Which minimum do we end up in? | sharpness, edge of stability, progressive sharpening, SAM, warmup |
| 11 | landscape | Why big networks are easier | Non-convex, yet training works | overparameterization, symmetry, spurious minima, benign landscape, NTK / lazy training, mean field |
| 12 | limits | How hard can it be? | What is provably the best possible? | oracle, upper/lower bound, Õ, NP-hard |

Test stage: shell + chapters 1–3 shown to Faris first (as with the Lee notes page), then chapters 4–12.
