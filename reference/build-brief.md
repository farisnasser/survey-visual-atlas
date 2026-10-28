# Survey Visual Atlas — Build Brief

Oct 28, 2026 · @Faris

## Goal

Build the **Survey Visual Atlas**: one web app that lists every paper in the MLR570 non-convex optimization survey and, for each paper, shows its 3-page math notes next to an interactive 3D visual of its core ideas.

The math notes explain *how* a result works. The visuals make the reader *see* it in under a minute: the saddle, the valley, the ball that escapes or gets stuck. Together they let Faris and Salah understand 30+ papers fast and compare them side by side.

This brief is written for Claude Code. Read it fully before writing any code.

## Context

The survey is "Non-Convex Optimization in Neural Networks" for MLR570 at AUS, written by Faris and Salah. Phase 1 covers 30 foundational papers. Later phases add narrower searches, so the paper count will keep growing.

| Item | Where it lives | Notes |
| --- | --- | --- |
| Paper database (single source of truth) | Google Sheet in Drive, under MLR570 Project / NonConvex Survey | One row per paper. Faris maintains it; Salah works from it. |
| Per-paper math notes | PDFs in the same Drive folder | About 3 pages each, math from first principles, generated per paper |
| Paper PDFs | Drive (exact folder to confirm) | Needed to extract theorems, figures and the paper's own examples |
| Survey write-up | Overleaf | Out of scope for this build, but the Atlas should make figures easy to reuse there |

The gap: the notes are dense math with no pictures, so understanding each paper quickly is hard. The Atlas closes that gap.

## What already exists

One finished visual sets the bar: **saddles.html**, built for Kawaguchi, "Deep Learning without Poor Local Minima" (NIPS 2016). Faris will drop the file into the repo as `reference/saddles.html`. Every new paper visual should feel like a sibling of it.

What it does:

- Full-screen Three.js scene (three@0.128.0 + OrbitControls from jsDelivr), dark theme, drag to orbit, pinch to zoom.
- Four tabs, one concept each: **Valleys** (bumpy landscape vs. the paper's landscape where every valley is global), **Good saddle** (one hidden layer, curves down along one eigen-direction), **Bad saddle** (two or more hidden layers, Hessian has no negative eigenvalue, flat plateau), **Race** (same nudge, same step size, both saddles side by side).
- A ball runs real gradient descent on the true loss, leaving a trail, with a live step and loss readout.
- Curvature lines along Hessian eigen-directions, colored by sign, with short λ labels. Global minima drawn as glowing curves.
- Every surface is the actual loss of a tiny network, so the picture is mathematically honest:

```latex
\text{strict saddle: } L(w_1,w_2) = (1 - w_1 w_2)^2, \qquad \nabla^2 L(0) = \begin{pmatrix} 0 & -2 \\ -2 & 0 \end{pmatrix}
```

```latex
\text{bad saddle: } L(w_1,w_2) = (1 - w_1 w_2^2)^2, \qquad \nabla^2 L(0) = 0
```

- Text is minimal: paper title and authors, tab names, one-sentence caption, tiny legend.

A second, flatter example also exists (an XOR space-warp dashboard). It shows the 2D style for papers whose idea is a mapping rather than a landscape.

## Requirements: the hub

The hub is the landing page: every paper in the survey at a glance, one click to its page.

**Must have**

- A card or row per paper, generated from the Google Sheet, never typed by hand. Shows title, authors, year, venue, survey phase, topic tags and status.
- A small animated thumbnail per paper taken from its visual (a still frame is fine at first).
- Status per paper, shown as a chip: *Notes only*, *Visual in progress*, *Complete*.
- Search by title, author or keyword. Filter by phase, topic tag, year and status.
- Grouping by survey theme (for example: loss landscape geometry, saddle escape, SGD dynamics, overparameterization, convergence guarantees). Themes come from a Sheet column.
- Progress summary at the top: papers total, notes done, visuals done.
- Works on a phone and a laptop.

**Nice to have**

- A timeline view by year, so the reader sees how ideas built on each other.
- A concept index: click "strict saddle" and see every paper that uses it, with a link to each visual tab.
- "Compare" mode: pick two papers and see their visuals side by side.
- A citation-graph view if the Sheet records which papers cite which.

## Requirements: the paper page

Each paper gets one page with four parts. The visual leads; the notes support it.

1. **Header.** Title, authors, year, venue, links to the original paper and the notes PDF, theme tags, and one sentence stating the paper's main result in plain words.
2. **Visual, the main event.** An interactive scene in the saddles.html style, one tab per core concept (usually 2 to 4). 3D by default. Use 2D charts where the idea is a curve, a distribution or a trajectory over time rather than a surface.
3. **Math, next to the picture it explains.** Each tab can open a short panel with the one to three equations behind it, rendered with KaTeX. It shows the definition, the key theorem statement and what the visual's toy model is. No derivations here; those stay in the notes.
4. **Notes.** The 3-page PDF embedded in the page with an in-page PDF viewer (PDF.js), plus a download link. Later, an optional HTML version of the notes so equations link to visual tabs.

**Charts where they help.** For stochastic or dynamic results, add 2D plots beside the 3D scene:

- loss vs. iteration, with several noisy SGD runs against one GD run
- gradient-noise histograms, or the step-size vs. batch-size trade-off
- Hessian eigenvalue spectrum at a chosen point
- convergence rate on a log scale, with the theorem's bound drawn as a reference line

**Controls the reader can change.** Learning rate, batch size or noise level, start point, depth or width, momentum on or off. Each control changes the live simulation, so the reader can break the paper's assumptions and see what happens.

**Text budget.** One sentence per tab caption, short labels on the scene, a small legend. No paragraphs on the visual itself.

## Visual standards

**Honesty rules (non-negotiable)**

- Every surface is a real function with the property the paper claims. Prefer the loss of a tiny network (2 to 3 scalar weights) or the paper's own example. Never hand-sculpt a shape to look right.
- Balls run the real optimizer (GD, SGD, momentum, Adam, perturbed GD) on the true loss. Display height may be rescaled (for example log(1 + L)), but the dynamics use the true L.
- Check every gradient and Hessian with a short Python script before building. Tune step sizes so the motion reads clearly (escape in about 1 s vs. 5 s, for instance).
- When a picture simplifies the theorem (2D slice of a high-dimensional space, scalar weights instead of matrices), say so in one line on that tab.

**Look and feel**

- Same shell for every paper: dark theme, header with title and tabs, caption card bottom-left, action button bottom-right.
- One shared color language across the whole Atlas: ball and trail coral, negative curvature red, positive curvature blue, flat or zero curvature amber, global minima green. Height ramp from deep indigo through teal to pale yellow.
- Shared visual library (`atlas3d.js`) so new papers reuse surfaces, balls, trails, eigen-lines, labels and charts instead of copying code.
- Phone-safe layout, `prefers-reduced-motion` respected, no external hosts beyond the pinned CDNs.

**Concept-to-visual catalog** (starting point; extend as papers need)

| Concept | Visual | Toy model |
| --- | --- | --- |
| Strict saddle | Saddle surface, red down-curve, ball slides off fast | (1 − w₁w₂)² |
| Degenerate (bad) saddle | Flat plateau, amber flat lines, ball creeps | (1 − w₁w₂²)² |
| Spurious local minimum | Bumpy surface, ball trapped in a shallow pit | quadratic + cosine bumps |
| All local minima global | Valley floor glowing green, balls from random starts all reach it | low-rank factorization loss |
| Escaping saddles with noise | Many SGD balls fanning out from a saddle vs. one stuck GD ball | strict saddle + Gaussian gradient noise |
| Sharp vs. flat minima | Two basins side by side, perturb weights and watch loss jump | two Gaussians of different widths |
| Step size and stability | Ball overshooting and diverging as the rate passes 2/λ\_max | quadratic bowl with a slider |
| Overparameterization | Same task at growing width, bumps smoothing out | random-feature model |
| Mode connectivity | Two minima joined by a low-loss path | the paper's path construction |
| Convergence rate | 2D log-loss vs. iteration with the theorem's bound | per paper |

## Data pipeline

Drive stays the source of truth. Claude Code only reads it and writes files into the repo; it never edits the Sheet or notes without asking.

&#91;embedded content: data pipeline · Drive to site\]

The top two rows are automatic and re-run on every sync, so a new Sheet row or notes PDF shows up on the hub without hand work. The bottom row is the creative step: Claude Code runs the paper-visuals skill per paper, and Faris reviews each result.

## Architecture and repo layout

A static site, no server: the build step reads Drive and writes plain files, and the site renders them. That keeps hosting free and every paper page shareable as a link.

- **Stack:** Vite + vanilla JS (or Astro if Claude Code prefers content collections), Three.js 0.128 for 3D, a small charting layer (Observable Plot or hand-rolled SVG) for 2D, KaTeX for equations, PDF.js for notes.
- **Hosting:** GitHub Pages or Netlify from a private repo. Confirm with Faris before anything goes public.
- **One folder per paper**, keyed by a stable slug from the Sheet, so a visual can be built and reviewed on its own.

```
survey-atlas/
  CLAUDE.md                  this brief, condensed; project rules
  reference/saddles.html     the bar every visual must meet
  data/papers.json           generated from the Sheet, never edited by hand
  scripts/sync-sheet.(js|py) Sheet -> papers.json
  scripts/sync-notes.(js|py) Drive notes PDFs -> public/notes/
  scripts/check-math/        Python checks per paper (gradients, Hessians, step sizes)
  src/lib/atlas3d.js         shared surfaces, balls, trails, eigen-lines, labels
  src/lib/charts.js          shared 2D plots
  src/hub/                   landing page
  src/paper/                 paper page shell (header, tabs, math panel, notes viewer)
  papers/<slug>/
    visual.js                the paper's tabs and scenes
    math.md                  equations shown beside each tab
    meta.json                hand-written extras: one-line result, toy model notes
  public/notes/<slug>.pdf    synced notes
```

- `papers/<slug>/visual.js` exports a list of tabs: name, caption, surfaces, optimizers, controls, charts. The shell renders it. This keeps each paper small and consistent.
- A paper with no `visual.js` yet still shows on the hub and its page shows the notes, marked *Notes only*.

## Build phases

Each phase ends with something Faris can open and react to. Do not start the next phase until he signs off.

1. **Read and inventory.** Read the Sheet and list its columns. List the notes PDFs and match each to a Sheet row. Report mismatches (rows with no notes, notes with no row). *Done when:* Faris confirms the column mapping and the match list.
2. **Skeleton.** Repo, `CLAUDE.md`, sync scripts, `papers.json`, hub listing all 30 papers with notes embedded. No visuals yet. *Done when:* every paper opens and shows its notes.
3. **Shared library.** Port saddles.html into `atlas3d.js` and the paper shell. Rebuild the Kawaguchi page on it with no loss of quality. *Done when:* the rebuilt page matches the reference side by side.
4. **Pilot batch of 3.** Faris picks 3 papers with different flavors (say, one landscape, one SGD dynamics, one convergence rate). Build each, including any 2D charts. *Done when:* Faris approves all three. Adjust the library and checklist from his feedback.
5. **Roll out.** Remaining papers in batches of 5, most-cited first, each through the checklist below. Update status in the hub.
6. **Cross-paper features.** Concept index, compare mode, timeline. Optional citation graph.
7. **Later phases of the survey.** New Sheet rows flow in through sync; each new paper gets notes, then a visual through the same pipeline.

## Per-paper checklist

Run this for every paper before marking it *Complete*.

- [ ] Read the paper PDF and its notes. List the 2 to 4 core claims (abstract, theorems, corollaries, figure captions).
- [ ] Each claim maps to one tab, with a named toy model or the paper's own example.
- [ ] Python check passes: gradients and Hessians match finite differences; the shown property holds (eigenvalue sign, minimum value, escape time).
- [ ] Optimizers run on the true loss; step sizes tuned so the motion is readable.
- [ ] Math panel per tab: definition, theorem statement in one line, the toy model's formula.
- [ ] Simplifications stated in one line where the picture differs from the theorem.
- [ ] Colors follow the shared language; text within budget.
- [ ] Works at phone width and on a laptop; reduced motion respected; no console errors.
- [ ] Notes PDF opens on the page; links to the paper and notes work.
- [ ] Hub entry updated: status, thumbnail, tags.

## What I need from Faris

**Files to hand over**

- [ ] `saddles.html` (from this chat) into `reference/`
- [ ] This brief, exported as Markdown, into the repo as the start of `CLAUDE.md`
- [ ] The paper-visuals skill (Appendix A) saved in Claude Code
- [ ] Links to the Sheet, the notes folder and the paper PDFs folder in Drive

**Questions to answer**

- [ ] Sheet columns: which ones exist today (title, authors, year, venue, link, phase, theme, notes filename)? Can we add `slug`, `theme`, `status` and `cites` columns?
- [ ] Are notes PDFs named consistently, or linked from the Sheet?
- [ ] Is the paper PDF itself in Drive for every row, or only a link?
- [ ] Who sees the Atlas: just you and Salah, the course instructor, or public? This decides hosting.
- [ ] Deadline for the survey, and when you need the pilot batch.
- [ ] Which 3 papers for the pilot?
- [ ] The blueprint for later survey phases, when you have it, so the hub's themes and filters fit what's coming.
- [ ] Should the Atlas also export still images (PNG/SVG) of visuals for the Overleaf write-up?

## Appendix A: paper-visuals skill

Save this as `.claude/skills/paper-visuals/SKILL.md` in the repo. It is the same workflow used for the Kawaguchi page, adapted to the Atlas.

```markdown
---
name: paper-visuals
description: Build the interactive visual for one survey paper in the Survey Visual Atlas. Use when asked to visualize a paper, add a paper's visual, or work on papers/<slug>.
---

# Paper -> interactive concept visual

Goal: one paper page whose visual makes the paper's main ideas visible in under a minute. Visual first, very little text. Must match reference/saddles.html in quality.

## Steps
1. Read the paper PDF (pdftotext -layout) and its notes PDF. Grep the abstract, theorems, corollaries, figure captions. Pick the 2-4 core claims.
2. For each claim, find a tiny honest model: a 2-3 parameter loss or function whose math really has the property (e.g. L=(1-w1*w2)^2 for a strict saddle; L=(1-w1*w2^2)^2 for a degenerate saddle with zero Hessian), or the paper's own example.
3. Write scripts/check-math/<slug>.py: verify gradients and Hessians against finite differences, confirm the property, and tune step sizes so the motion reads clearly.
4. Write papers/<slug>/visual.js using src/lib/atlas3d.js and src/lib/charts.js. One tab per claim:
   - height-colored surface with faint grid lines
   - ball(s) running the real optimizer on the true loss, with trails
   - eigen-direction curvature lines colored by sign, short lambda labels
   - side-by-side race/contrast tab when the paper compares cases
   - 2D charts for stochastic or rate results (loss vs. iteration, noise, spectra)
   - one action button per tab, live step + loss readout, sliders for lr / noise / depth where relevant
5. Write papers/<slug>/math.md: per tab, the definition, the theorem in one line, the toy model formula (KaTeX).
6. Text budget: tab names, one-sentence caption per tab, tiny legend. State any simplification in one line.
7. Shared colors: ball coral, negative curvature red, positive blue, flat amber, global minima green.
8. Run the per-paper checklist in CLAUDE.md, then update the paper's status in the hub.
```

## Appendix B: kickoff prompt for Claude Code

Paste this as the first message, with this brief and saddles.html in the repo.

```markdown
We're building the Survey Visual Atlas for my MLR570 survey on non-convex optimization in neural networks. The full brief is in CLAUDE.md; the quality bar is reference/saddles.html. Read both fully before doing anything.

You have Google Drive access. The paper database is the Google Sheet in MLR570 Project / NonConvex Survey, and the per-paper notes PDFs are in the same folder.

Start with Phase 1 only:
1. Read the Sheet and list its columns with 3 sample rows.
2. List the notes PDFs and match each to a Sheet row. Show unmatched items on both sides.
3. Propose the column mapping and any columns to add (slug, theme, status, cites).
4. Propose the stack and repo layout, flagging anything you'd change from the brief and why.

Stop after that and wait for my sign-off. Don't write app code yet. Don't change the Sheet without asking.
```
