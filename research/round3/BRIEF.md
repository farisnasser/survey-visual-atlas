# Round 3 build brief (shared by sessions A, B and C)

Goal: one interactive page for every paper assigned to your session, at the same quality bar as the 35 existing pages.
The coordinator (Faris's main chat) merges the three branches, regenerates shared files, audits and pushes.

## Read first
- `CLAUDE.md` (project rules: real functions, real optimizers, check every formula, colour language)
- `.claude/skills/paper-visuals/SKILL.md` (tab API, workflow, the "at least one 3D tab" rule)
- `research/findings.md` (how verified findings are written) and two or three existing pages, e.g.
  `papers/jin2018accelerated/`, `papers/bhojanapalli2016global/`, `papers/gurbuzbalaban2021heavy/` and their
  `scripts/check-math/<key>.py`
- `data/papers.json` for each paper's survey number `n`, title, authors, branch and citing sections

## Per paper (repeat for each key on your list)
1. **Notes PDF.** `public/notes/<key>.pdf`. If missing (list in your prompt), download it from Drive with the
   `download_file_content` tool, **one file per call, never in parallel** (parallel calls overwrite each other's
   temporary result file). Decode with `python scripts/decode_mcp.py <tool-results dir> .cache/notes3`, check the
   decoded file starts with `%PDF` and has the expected Drive title, then copy it to `public/notes/<key>.pdf`.
   Drive is read-only: never edit, move or delete anything there.
2. **Read the original paper online** (arXiv / PMLR / publisher / author copy), not just the notes. Read the
   abstract, every theorem/definition statement with assumptions, figure captions and the experimental setup.
   Trust the notes' math but verify every claim you put on a page against the paper. Note anything the survey
   (its reference text in `data/survey.json`) or the notes get wrong.
3. **Check script** `scripts/check-math/<key>.py` using `common.py`; it must end with `done()` and pass. Verify
   gradients/Hessians by finite differences, confirm every number the page will state, write precomputed data to
   `papers/<key>/data.json` when needed. If a check contradicts the plan, change the visual, not the check.
4. **Page** `papers/<key>/visual.js`, `math.md`, `meta.json` (2–3 tabs, at least one 3D, honest captions and notes;
   reported-not-reproduced numbers are labelled as such). Surveys, monographs and textbooks get 1–2 tabs showing
   their central idea or the one result the survey cites them for.
5. **Audit** on your own local server: start `python scripts/serve.py <PORT>` with Bash `run_in_background`
   (your port is in your prompt), run `python scripts/sync.py` (local only, see below), open
   `http://localhost:<PORT>/paper.html?id=<key>` in the browser pane, click every tab, use
   `window.__atlas.tick(n)` and confirm `window.__atlasErrors` is empty, readouts match the check script, and no
   horizontal overflow at 375 px width.
6. **Thumbnail**: `http://localhost:<PORT>/thumbs.html?only=<key1>,<key2>` (writes `public/thumbs/<key>.webp`).
7. **Findings**: append a section per paper to `research/round3/<SESSION>-findings.md` in the style of
   `research/findings.md` (title line with `[n]`, verified statements with theorem numbers, notes' examples
   reproduced, survey/notes discrepancies, the planned/built visuals).
8. **Links**: add the official paper page and PDF URLs to `research/round3/<SESSION>-links.json` as
   `{"<n>": {"link": "...", "pdf": "..."}}` (check they load).

## Do not touch (shared files the coordinator regenerates or merges)
`data/*.json` (including `data/papers.json`: you may regenerate it locally with `scripts/sync.py` for testing, but
run `git checkout data/papers.json` before committing), `research/findings.md`, `research/v2-plan.md`, `src/`,
`index.html`, `paper.html`, `CLAUDE.md`, `.claude/`, `scripts/*.py` other than your own check scripts,
`scripts/check-math/common.py`, `scripts/check-math/results.txt`. If a shared fix is truly needed, describe it in
your findings file under "Shared changes needed" instead of making it.

## Commit, do not push
Commit to your own branch after every 3–4 finished papers (message: the keys added). Never push, never merge,
never switch branches. At the end run `python scripts/check-math/run_all.py` only for your own scripts (or run
each of yours individually) and report: papers built, any that could not be completed and why, every survey or
notes error found, and any shared change needed.
