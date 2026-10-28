# Survey Visual Atlas

Interactive companion to our MLR570 survey, *Non-Convex Optimization in Neural Networks: A Survey of Critical Points, Training Dynamics, and Escape Methods* (Faris Nasser & Salah Moussa, American University of Sharjah, Fall 2026).

**Live site: https://farisnasser.github.io/survey-visual-atlas/**

Every paper in the survey gets a page with:
- an interactive 3D/2D visual of its core results (drag to orbit, sliders to break the assumptions),
- the math behind each view (definitions, theorem statements, the toy model), rendered with KaTeX,
- our study-notes PDF.

## Honesty
Every surface is a real function — the loss of a tiny network, the paper's own example, or a slice of a real network's loss — and every ball runs a real optimizer (GD, SGD, Newton, saddle-free Newton, perturbed GD, line searches) on it. Display heights are rescaled; dynamics are not. Each paper's formulas, step sizes and replicated experiments are verified in [`scripts/check-math/`](scripts/check-math) (`python scripts/check-math/run_all.py`); [`research/findings.md`](research/findings.md) records what each paper actually proves, checked against the PDF. Where a picture simplifies a theorem, the page says so.

## How it works
Static site, no build step. The paper list is generated from the team's Papers Master sheet (`scripts/sync.py`), the notes come from the shared Drive folder, and each paper lives in `papers/<Key>/` (`visual.js`, `math.md`, `meta.json`). Shared code: `src/lib/atlas3d.js` (Three.js scenes), `src/lib/charts.js` (2D plots). Run locally with `python scripts/serve.py` → http://localhost:8570.

Built with Claude Code. Paper summaries in the tracker are AI-drafted and team-edited; theorem statements on the pages were checked against the original papers.
