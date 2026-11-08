# Survey guide (v2): a visual journey through our survey, section by section

Brief (Faris, 2026-11-08): rebuild the Atlas to suit the new survey
(`MLR570/survey-v2/`, *Gradient Dynamics from Saddle Escape to the Edge of Stability*).
- **Not paper by paper**, because papers overlap. Instead, follow the survey section by section.
- Explain the math preliminaries from essential principles.
- Include all the visuals needed (balls, curves).
- **Every graph gets an explanation of what we are looking at.** Nothing vague.
- When done: push to the same repo, `farisnasser/survey-visual-atlas`.

## Shape
- `index.html` becomes the guide's map: the 16 chapters grouped by survey section.
- `guide.html?sec=<id>` shows one chapter.
- The old paper hub moves to `library.html` (kept, labelled "first survey's paper library"). Paper pages stay, and their brand link goes to `library.html`.
- `learn.html` (the crash course) stays. Its brand link goes back to `./`.
- Files: `guide/guide.json` (chapters: id, sec, n, title, question, terms), `guide/<id>/text.md`, `guide/<id>/viz.js`, `src/guide/guide.js` (clone of `learn.js`), `src/guide/guide.css`.
- Terms come from `course/glossary.json` (96 terms), plus `guide/glossary.json` for terms new to the guide. Popover links go to the chapter that builds the term (guide chapter or crash-course stop).
- New box kind `:::read How to read this visual`. It follows **every** `:::viz` and covers: axes, colours, what moves, what to try, what to notice.
- Reused visuals: `guide/<id>/viz.js` imports crash-course stops (`course/<stop>/viz.js`, default export = name → tab) and paper tabs (`papers/<Key>/visual.js`, `.tabs`). Tabs that need `data.json` get `data` attached. Widget sets `ctx.data`, `ctx.base` and `ctx.shell = { tab }`.
- New visuals get `scripts/check-math/guide_<id>.py`.

## Chapters (survey section → guide chapter → visuals; R = reused, N = new)
| # | id | Survey § | Title | Visuals |
|---|---|---|---|---|
| 1 | intro | I | The puzzle and the plan | R goal.valleys · N twofaces-teaser |
| 2 | loss | II-A,B | From a network to a loss, and its local shape | R goal.fit · R landscape.mirror · R slope.compass · R slope.plane · R curvature.quad |
| 3 | points | II-C,D | Flat points, success, and assumptions | R critical.zoo · R critical.sosp · R steps.lipschitz · R steps.upper · R noise.cloud |
| 4 | algos | II-E | The algorithms under study | R steps.parabola · R noise.floor · R memory.valley · R adaptive.precond · R escape.newton |
| 5 | multiplier | II-F | One multiplier: linear stability and 2/η | N multiplier · R curvature.zigzag · R escape.escape |
| 6 | avoid | III-A,B | Why saddles matter; GD almost never stops at one | R critical.random · R kawaguchi.bad · R lee.stable · R lee.map · R mertikopoulos.ridge |
| 7 | escape | III-C,D | Avoidance is not efficiency; noise | R du.chain · R jin.kick · R jin.pancake · R daneshmand.cnc · R gurbuzbalaban.tail · R azizian.camel |
| 8 | stages | III-E | Saddles as stages of learning | R saxe.plateaus · R saxe.hyperbolas · R zhang.manifold · R zhang.loss |
| 9 | precursors | IV-A,B | Before the edge: stability of SGD, break-even, catapults | N sgdstab · R lewkowycz.spike · R lewkowycz.phases |
| 10 | edge | IV-C | The edge of stability | R cohen.quad · R cohen.mlp · R cohen.mom · R sharpness.wells |
| 11 | why | IV-D | Why it does not blow up | R damian.stages · R damian.orbit · R wang.trace · N normgd · R chen.bif · R kalra.chaos |
| 12 | flow | IV-E | Central flows | N centralflow |
| 13 | beyond | IV-F,G | Adam, warmup, SAM, SGD; large steps change the destination | R kalra24.adam · R kalra24.warm · R agarwala.edge · R mohtashami.escape · R galli.flatland |
| 14 | compare | V | One instability, two faces | N twofaces |
| 15 | open | VI, VII | Open problems and the take-away | R kawaguchi.race · R gurbuzbalaban.tail(alpha) · R saxe.lu0 |

## Rules (same as the rest of the Atlas)
- Every claim matches the survey v2 text and its verification log (`../survey-v2/verification/claims.md`) or `research/findings.md`.
- Every surface is a real function and every ball runs a real optimizer. Every number comes from a check script or the paper.
- The tone is the crash course's: short declarative sentences, "we", define every symbol at first use, no hype.

## Status (2026-11-08)
All 16 chapters are written and live. Chapter 6 (dynamics: survey II-G/H/I, GD as a map, noise, measuring curvature) was added after the survey's first-principles pass, which also renumbered the survey's equations; the guide's equation and chapter references were renumbered with scripts/renumber_guide_eqs.py and renumber_guide_chapters.py.
- **Visuals:** 67 inline visuals, plus 7 of the survey's own figures, each with a "How to read" box.
- **New visuals (4), check-scripted:** multiplier (`guide_multiplier.py`), sgdstab (`guide_precursors.py`), normgd (`guide_why.py`), centralflow (`guide_flow.py`). `guide_terms.py` checks that every term resolves, every anchor exists, and every visual and figure has its read box.
- **Reused visuals:** the rest come from crash-course stops and paper tabs. `damian2023self` and `kalra2025universal` now listen on `ctx.plot.c` instead of `#plot2d`, so they work inside chapters too.
- **Batch sharpness** has no live visual. That needs a non-linear SGD replication, which we have not check-scripted. Chapter 13 says so.
