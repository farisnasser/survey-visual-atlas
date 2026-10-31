---
name: crash-course
description: Write or extend a stop of the Atlas crash course (learn.html, course/<id>/text.md + viz.js + scripts/check-math/course_<id>.py, glossary terms). Use when asked to add or improve a crash-course chapter, its visuals, or glossary terms.
---

# Crash course: one stop of the journey

Faris's brief: a part of the Atlas that introduces "all the terms in optimization, with the proper visuals, and basically
everything you need to digest the papers in the Atlas", told "like a journey" (we want to minimize the loss; then this
problem appears, so people came up with that fix), math "built from first principles", "every weird new term or variable
defined and explained in simple words", "the ultimate guide for anyone wanting to understand optimization and learn what the
field is like". Stops 1–3 (`course/goal`, `course/slope`, `course/steps`) are approved: **read all three (text, viz, check
script) before writing anything**; every new stop must read like their sibling. The plan of all 12 stops is in
`research/course-plan.md`.

## Files of a stop
- `course/<id>/text.md` — the article. Syntax = the in-page notes syntax (`src/paper/notes.js`): `##` numbered sections
  (`## 3. Title`), `###`, `$…$` / `$$…$$` (KaTeX 0.16.9), lists, pipe tables (no bare `|` inside cells: use `\lVert`, `\lvert`,
  `\mid`), `:::kind Title … :::` boxes, `:::cols`, `:::figure`, plus the course additions:
  - `:::viz <name>` + `:::` — a live visual slot, mounted from `course/<id>/viz.js` export `<name>`.
  - `[[term-id]]` or `[[term-id|shown text]]` — a glossary term (dotted underline, hover/tap popover).
  - `[text](paper.html?id=<key>)` — link to an Atlas paper page (keys from `data/papers.json`; check each exists).
  - Box kinds used by the course: `def` (definition, amber), `key` (the idea to keep), `theorem` (a paper's result, with
    theorem number), `warn` (common confusion / what is *not* proved), `idea` (the fix people came up with), `problem`
    (the problem that opens the next stop), `atlas` ("Where this shows up in the Atlas").
- `course/<id>/viz.js` — `export default { name1: tab, name2: tab, … }`. Same tab API as paper visuals (see the
  `paper-visuals` skill and `src/learn/widget.js`): `{ title, caption, note, legend, mode: '3d' | '2d', cam, tgt, plot,
  controls, actions | action + drop, charts, build(ctx), frame(ctx, dt), onControl(ctx, id, v) }`. 3D uses
  `src/lib/atlas3d.js` (`def`, `ctx.stage.surface/ball/marker/eigenLines/label/dot/tube/line`, `opt.gd/momentum/sgd/curv`);
  2D uses `ctx.plot` (`src/lib/charts.js`, `equal: true` for contour maps) and helpers in `src/learn/plot2d.js`
  (`contours`, `drawContours`, `heat`, `arrow`, `draggable`). Side charts via `charts: [...]` and `ctx.charts[id]`.
- `scripts/check-math/course_<id>.py` — recomputes **every number** the text, the visuals' readouts, notes and glossary state.
  Uses `common.py`, ends with `done()`, must pass.
- `course/course.json` — set `"ready": true` for the stop when it is finished (and keep its `terms` list = the glossary ids
  introduced in that stop).
- `course/glossary.json` — every term the stop introduces has an entry `{term, ch, def, tex?, aka?, sec}`; `sec` is the
  anchor of the section that builds it (`n-` + slug of the heading: lower-case, every run of non `a-z0-9` → `-`, e.g.
  `## 4. The descent lemma: why $\eta = 1/L$ is safe` → `n-4-the-descent-lemma-why-eta-1-l-is-safe`). Entries for all 12
  stops already exist (written before the stops); refine a definition only if the stop shows it is imprecise, add new terms
  your stop needs, and keep entries grouped by stop (they are in stop order in the file).

## How a stop is written (the journey)
1. **Open with the problem** the previous stop ended on (its `:::problem` box), in one or two sentences.
2. **Build the math from first principles**, smallest example first (1-D, then 2-D, then $d$-D). Derive when the derivation
   is short (a few lines); otherwise state the result precisely, say who proved it (paper + theorem number from
   `research/findings.md`), and say that we do not prove it here.
3. **Define every symbol at first use** in plain words ("$\kappa$ (kappa), the condition number, is …"). Link every term
   with `[[id]]` at the place it is defined, and the first time it is used in the stop.
4. **Worked numbers**: at least one small concrete example per main idea, with numbers that the check script reproduces.
5. **Visuals between the paragraphs** (3–5 per stop, at least one 3D): each makes one idea visible and interactive. Put the
   visual right after the paragraph it illustrates.
6. **Name the fixes people came up with** (`:::idea`), and what each costs or does not guarantee (`:::warn`).
7. End with `:::atlas Where this shows up in the Atlas` (papers that use or prove the ideas, linked) and `:::problem The next
   problem` that opens the next stop (stop 12 ends with where to go next in the Atlas instead).

Tone (learn it from stops 1–3): short declarative sentences, present tense, "we" for the reader and author together, bold the
term where it is defined, concrete before abstract, no hype, no "simply"/"obviously"/"it is easy to see", no rhetorical
questions in a row, no emoji. Separate clearly what is **proved** (and under which assumptions) from intuition and from what
is only **observed** in experiments.

Notation (from stop 1, section 6): minimize $f(x)$, $x \in \mathbb{R}^d$, iterates $x_t$, step size $\eta$, $f^\star = \inf f$,
gradient $\nabla f$, Hessian $\nabla^2 f$ (or $H$), eigenvalues $\lambda_{\min}, \lambda_{\max}$, smoothness $L$, strong
convexity $\mu$, condition number $\kappa = L/\mu$, Hessian-Lipschitz $\rho$, tolerance $\varepsilon$, momentum $\beta$, noise
variance $\sigma^2$, batch size $B$, data size $n$. If a paper uses other letters, say so once.

## Accuracy (non-negotiable)
- Every claim about a paper (what it proves, rates, assumptions, theorem numbers, what it observes) must match
  `research/findings.md` (verified against the PDFs) or the paper itself, read now. Rates always with their target
  (ε-FOSP vs ε-SOSP), oracle (deterministic / stochastic gradients, Hessian-vector products) and hidden factors.
- Every number on the page (text, readouts, captions, notes, glossary) comes from the check script or the paper.
- Every surface is a real function and every ball runs a real optimizer; display height may be rescaled (say so in the
  `note`), dynamics never. If a picture simplifies a theorem (2-D picture of a $d$-D fact), say so in one clause.
- Standard facts (definitions, textbook identities) stated exactly, with conditions.

## Visual quality checklist
- Real interaction: a slider or a drag that changes the math, plus a readout (`ctx.stat`) with the numbers that matter.
- Colour language (never change): optimizer 1 coral `#ff7a59`, 2 violet `#b28cff`, 3 white; negative curvature red
  `#ff5d73`, positive blue `#7cc4ff`, flat amber `#ffc857`, global minima green `#5fe0b0`.
- Text budget: title, one- or two-sentence caption, one-line note, small legend. Explanations live in the article.
- **Framing**: screenshot every 3D visual. The key object (ball path, saddle, plane) must be visible, the surface not
  clipped at the top, steep walls not blocking the view (narrow the domain with `square: true`, move the camera, lower
  `fit`). Stops 1–3 needed this twice.
- 2D contour maps need `equal: true` so perpendicularity and circles look right.

## Audit before calling a stop done
1. `python scripts/check-math/course_<id>.py` passes.
2. `python scripts/serve.py <PORT>` (background), open `http://localhost:<PORT>/learn.html?ch=<id>`. A hidden browser pane
   never fires intersection callbacks, so run `window.__course.mountAll()`, then for each widget
   `window.__course.widgets[name].tick(300)` (click its first action first if it has one). Require
   `window.__atlasErrors` empty, `document.querySelectorAll('.katex-error').length === 0`, and readouts equal to the check
   script. Use `resize_window` (1280×900, then mobile) before measuring; `document.documentElement.scrollWidth - innerWidth`
   must be ≤ 0 at 375 px.
3. `learn.html?ch=glossary` renders with no KaTeX errors; every `[[id]]` in the stop resolves (a quick script:
   all ids in `course/*/text.md` are keys of `course/glossary.json`).
4. Screenshots of every visual at desktop width look right (framing, legibility, nothing overlapping).
