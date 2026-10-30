---
name: paper-visuals
description: Build the interactive visual page for one survey paper in the Survey Visual Atlas (papers/<Key>/visual.js, math.md, meta.json, check script). Use when asked to visualize a paper, add a paper's visual, or work on papers/<Key>.
---

# Paper → interactive concept visual

Goal: a page whose visual makes the paper's main results visible in under a minute. Visual first, very little text.
It must feel like a sibling of `reference/saddles.html` and of the existing pages (open two or three before starting).

## 1. Read
- Paper PDF: `pdftotext -layout` it; read the abstract, every theorem/definition/corollary statement, the figure
  captions and the experimental setup. Notes PDF: read all of it.
- Pick 2–4 core claims. Write them, with equation/theorem numbers, exact assumptions and the survey reference number `[n]`, into `research/findings.md`. Note anything the survey or the notes state differently from the paper.

## 2. One honest toy model per claim
Use, in order of preference: the paper's own example or experiment; a 2–3 parameter loss that provably has the
property (e.g. (1 − w₁w₂)² strict saddle, (1 − w₁w₂²)² degenerate saddle); a slice of a real network's loss
(say which slice); a live or offline replication of the paper's experiment. Numbers reported by the paper but not
reproducible here may be shown as "reported in the paper" — never as computed.

## 3. Check the math first — `scripts/check-math/<Key>.py`
Use `common.py` (`check_grad`, `check_hess_at`, `eig`, `gd`, `save_json`, `sig`). Verify gradients/Hessians against
finite differences, confirm the property (eigenvalue signs, minima values, escape counts), tune step sizes, and write
any precomputed data to `papers/<Key>/data.json` (round losses with `sig`, not `r`). It must end with `done()` and pass.
If a check contradicts what you planned to show, change the visual, not the check.

## 4. `papers/<Key>/visual.js`
```js
export default { tabs: [ {
  id, name, title, caption /* one sentence */, note /* one line: simplification or setup */,
  legend: [[hex, label]], mode: '3d' | '2d', cam: [x,y,z], tgt: [x,y,z],
  controls: [{ id, label, min, max, step, value, fmt, live }] or { type: 'seg', options: [[v, label]] },
  action: 'Drop balls'  |  actions: [{ label, fn(ctx) }],
  charts: [{ id, title, xlabel, ylabel, logx, logy, xlim, ylim, h }],   // side panel, ctx.charts[id] is a Plot
  plot: { ... },                                                           // 2D tabs: ctx.plot options
  build(ctx), drop(ctx), frame(ctx, dt), onControl(ctx, id, v)
} ] }
```
- 3D: `def({ L, g, hess?, h, dom, fit, square })`, `ctx.stage.surface(d)`, `ball(d, { start, step: opt.gd(lr), stop })`,
  `eigenLines(d, x, y)`, `tube`, `line`, `marker`, `label`, `points`; `opt.sgd / momentum / curv(lr, 'newton'|'sfn')`
  or a custom `step(p, ball, d)`. Live readout with `ctx.stat(text)` in `frame`.
- 2D: `ctx.plot.clear()...line/fn/scatter/hline/vline/heat/arrows/bars/custom...draw()`.
- Colours: coral ball, violet/white for extra optimizers, red λ<0, blue λ>0, amber flat, green global minima.
- Text budget: tab names, one-sentence caption, one-line note, tiny legend. No paragraphs on the visual.
- At least one tab must be 3D: thumbnails render the WebGL stage only (`meta.thumb` names it; `meta.thumbTicks` sets how many frames run first).
- Rebuilding a 3D tab in `onControl` with `ctx.stage.clear(); this.build(ctx)` is fine: `clear()` keeps the tab's frame callback.
- Live MC/randomness: seed it (`rng(seed)`) when the stat line quotes a number that should match the check script.

## 5. `math.md` and `meta.json`
`math.md`: one `## <tabId> | Title` section per tab — definition, the theorem in one or two lines with its number,
and the toy model's formula; `> ` lines for check-script numbers. `meta.json`: `result` (one sentence, may use <b>),
`result_plain`, `tags` (reuse existing concept names where they fit: see the hub's Concepts view), `tag_tabs`,
`tabs`, `thumb` (a 3D tab), `data: true` if it loads `data.json`.

## 6. Check before calling it done
Localhost audit of every tab (`window.__atlas.tick(n)`, `window.__atlasErrors` empty, readouts match the check
script), screenshot at laptop and phone width, thumbnail regenerated, `scripts/sync.py` re-run, hub shows Complete.
