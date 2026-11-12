# Survey Visual Atlas — project rules

Since 2026-11-10 the home page is **the guide to survey v3** (`MLR570/survey-v3/`, *From Gradients to Curvature and Matrix Geometry: A Survey of Optimization Methods for Deep Neural Networks*): 18 chapters following its Sections I–VIII with the appendices folded in, every visual followed by a "How to read this visual" box, the survey's own 14 figures as `:::survey` blocks, and a live dashboard of the real Fig. 9 runs. On 2026-11-12 it was aligned with the **final, rewritten** survey (36 pages, 72 references, 66 equations): survey equation numbers were remapped with `scripts/survey3_renumber_map.py` (old → new map in `research/survey3_renumber.json`) and `scripts/renumber_guide3_eqs.py`; references to equations the rewrite dropped now point to the survey section that describes them in words (`scripts/fix_gone.py`); the 3D survey figures were recropped and their "How to read" boxes rewritten; three visuals were added ('narrow' and 'zoo' in chapter 4, 'memory' in chapter 17). If the survey is renumbered again, rebuild the map and rerun the renumber script. The previous guide (survey v2, saddle escape to the edge of stability, 16 chapters) lives in `guide-v2/` and is served at `guide.html?v=2` (map) and `guide.html?v=2&sec=<id>`; `src/guide/guide.js` switches on `?v=2`. The original per-paper Atlas (every reference of the first, broader survey, organized by its Fig. 5 taxonomy, with an interactive visual, the math and the study notes per paper) is kept as the **paper library** (`library.html`). Live at **https://farisnasser.github.io/survey-visual-atlas/** (GitHub Pages, public repo `farisnasser/survey-visual-atlas`, branch `main`, root).

Full original brief: `reference/build-brief.md`. Quality bar: `reference/saddles.html` (the Kawaguchi visual every page must feel like a sibling of).

## The one rule that matters
**Every surface is a real function and every ball runs a real optimizer on it.** Display height may be rescaled (`def.h`, `fit`), dynamics never are. Each paper's formulas are checked in `scripts/check-math/<Key>.py` before they go into `visual.js`; when a picture simplifies the theorem, the tab's `note` says so in one line. Never sculpt a shape to look right. Never state a number on a page that a check script or the paper didn't produce.

## Sources of truth (read-only)
- **The survey PDF** (`NonConvex_Survey_Final.pdf`, 111 references) decides which papers the Atlas lists, their numbering and grouping. `scripts/build_survey.py` turns it into `data/survey.json` (references, where each is cited, Fig. 5 taxonomy). Rebuild only when the survey changes.
- **Drive notes folder**, alumni account `b00097313@alumni.aus.edu`, `My Drive / MLR570 Project / NonConvex Survey / PDFs / Notes` (`1XmjqySjmFT-BPS04Uti5m6-ibsx0makK`). Notes file names follow no pattern; map them by hand in `data/notes_map.json` after reading the title page.
- `data/atlas_refs.json` (hand-maintained): survey reference number → Atlas key + official links.
- The old Papers Master sheet (`1DzVJJDWxBM6I5YbaMa21Yr5rlE5duIX5dK-sWnmGS94`) is no longer read; its AI summaries are frozen in `data/legacy_sheet.json`. Its column O "Our notes" was never published and must not be.
Never edit Drive.

## When Faris says new papers / notes were added
Run the **atlas-sync** skill (`.claude/skills/atlas-sync/SKILL.md`). In short: list the Drive notes folder, download new notes, map each to its survey reference number (`data/atlas_refs.json`, `data/notes_map.json`), `scripts/sync.py --notes-dir`, then for every paper the sync reports as "notes but no visual yet" run the **paper-visuals** skill (`.claude/skills/paper-visuals/SKILL.md`), regenerate thumbnails, commit, push. No scheduled or automatic refreshes — only on request.

## Layout
```
index.html                  the guide's map (src/guide/guide.js) · guide.html?sec=<id> one chapter · ?v=2 selects guide-v2/
guide/guide.json            survey-v3 chapters (id, survey section, title, question, terms, ready) · guide/glossary.json (built by scripts/build_guide3_glossary.py)
guide-v2/                  the survey-v2 guide, same structure (frozen)
guide/compare/data.json    measured Fig. 9 runs, written by scripts/check-math/guide3_numbers.py from ../survey-v3/experiments/results.json
guide/<id>/text.md|viz.js   one chapter: notes syntax + :::viz + :::read (How to read) + :::survey (our survey's own figures)
guide/lib.js                paperTab(key, id): reuse a papers/<Key>/visual.js tab inside a chapter (data.json attached)
public/survey3/figN.webp    survey v3's figures + its PDF (scripts/crop_survey3.py recrops and recopies from ../survey-v3/) · public/survey/ the same for survey v2
scripts/audit_pages.js      rendering audit on localhost: __audit(urls, width) for guide/learn pages, __auditPapers(keys) for every paper tab · scripts/check_paths.py exact-case path check (Pages is case-sensitive)
scripts/check-math/guide3_numbers.py   every number of the v3 guide's new visuals and texts (+ dashboard data) · guide_*.py the v2 guide's · guide_terms.py checks terms, anchors and read boxes of both guides
library.html, paper.html, compare.html, thumbs.html (local maintenance page)   the paper library (first survey)
src/theme.css               shared tokens + colour language
src/lib/atlas3d.js          surfaces (def/gridDef), optimizer balls (opt.gd/sgd/momentum/curv), eigen-lines, labels, points
src/lib/charts.js           canvas Plot: lines, log axes, reference lines, heat maps, arrows, bars
src/lib/math.js             seeded rng, Jacobi eigensolver, finite-difference checks
src/paper/paper.js|css      paper shell: header, tabs, HUD, side charts, KaTeX math drawer, PDF.js notes
src/hub/hub.js|css          hub: progress, search, filters, Survey map (Fig. 5) / References / Timeline / Concepts, compare
papers/<Key>/visual.js      { tabs: [...] } — see the paper-visuals skill for the tab API
papers/<Key>/math.md        "## <tabId> | Title" sections, $$display$$, $inline$, ### heads, - bullets, > source lines
papers/<Key>/meta.json      result (HTML), result_plain, tags, tag_tabs, tabs, thumb, data
papers/<Key>/data.json      only when a paper needs precomputed data (written by its check script)
learn.html                  crash course: journey map · ?ch=<id> one stop · ?ch=glossary (see the crash-course skill)
course/course.json          the 12 stops (order, titles, questions, terms, ready) · course/glossary.json every term
course/<id>/text.md|viz.js  one stop: article (notes syntax + :::viz slots + [[term]] links) and its live visuals
src/learn/                  learn.js (pages, term popovers), widget.js (inline visual = one paper-style tab), plot2d.js
scripts/check-math/course_<id>.py   every number of a crash-course stop
papers/<Key>/notes.md       in-page study notes (one-minute box, primer, notes verbatim) — see the study-notes skill;
                            rendered by src/paper/notes.js in place of the PDF viewer, PDF stays downloadable
data/survey.json            GENERATED by scripts/build_survey.py from the survey PDF
data/atlas_refs.json        survey ref number -> Key + links (hand-maintained)
data/notes_map.json         Key -> notes file name + Drive id (hand-maintained)
data/legacy_sheet.json      frozen summaries from the old sheet
data/papers.json            GENERATED by scripts/sync.py — never edit by hand
public/notes/<Key>.pdf      synced notes · public/thumbs/<Key>.webp generated thumbnails
scripts/build_survey.py     survey PDF -> data/survey.json
scripts/sync.py             survey.json + maps + notes folder -> papers.json + public/notes
scripts/decode_mcp.py       decode Drive-connector downloads (base64 JSON) into files
scripts/serve.py            dev server on :8570 + POST /__thumb for thumbnails
scripts/check-math/*.py     per-paper math checks (run_all.py runs every one)
research/findings.md        what each paper actually proves, checked against the PDF — read before touching a paper
```

## Colour language (never change per paper)
ball/optimizer 1 coral `#ff7a59` · optimizer 2 violet `#b28cff` · optimizer 3 white `#f2f4ff` · negative curvature red `#ff5d73` · positive curvature blue `#7cc4ff` · flat/zero curvature amber `#ffc857` · global minima green `#5fe0b0`. Height ramp indigo → teal → pale yellow. Dark theme only.

## Status on the hub (computed, not typed)
Complete = notes + visual · Notes only · Visual only (Kawaguchi: visual ported from the reference, no notes yet) · Not started = neither (shown as compact rows).

## Dev
`python scripts/serve.py` → http://localhost:8570 (also registered as preview "survey-atlas" in `faris-brain/.claude/launch.json`). On localhost the paper shell exposes `window.__atlas.tick(n)` (advance n frames without requestAnimationFrame) and `window.__atlasErrors`; use them to audit every tab. No build step: plain ES modules + pinned CDNs (three@0.128.0, katex@0.16.9, pdf.js 3.11.174).

## Deploy
Commit to `main` and push; GitHub Pages serves the repo root (`.nojekyll` present). Confirm the live URL loads after a push.
