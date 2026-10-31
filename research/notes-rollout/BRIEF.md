# In-page notes rollout brief (shared by sessions A–F)

Goal: replace the notes PDF viewer with in-page study notes on every paper assigned to your session, at the quality of
`papers/lee2016gradient/notes.md`. Faris's words: start each paper with "the paper in one minute", then a first-principles
explanation an ML master's student can digest, then "exactly what was in the notes … (EXACTLY) in terms of the math or any
graph", changing the notes "IF and ONLY IF" the paper shows they are wrong; "make sure there are no inaccuracies
whatsoever", "do not change much unless you feel completely necessary", "keep the flow of explanation similar everywhere".
The coordinator (Faris's main chat) merges the six branches, audits and pushes.

## Read first
- `CLAUDE.md` (project rules) and **`.claude/skills/study-notes/SKILL.md` — the full spec; follow it exactly.**
- `papers/lee2016gradient/notes.md` next to `public/notes/lee2016gradient.pdf` (render the PDF pages and compare: that is
  the transcription standard), and `scripts/check-math/lee2016gradient.py` (how the notes' numbers are checked).
- `research/findings.md`: what was already verified about each of your papers (theorem numbers, known notes/survey errors).

## Per paper (repeat for each key on your list, in order)
1. Render and read the notes PDF `public/notes/<key>.pdf` (all pages, as images) and extract its text.
2. Read the original paper (official PDF/link in `data/papers.json`). Verify every citation, statement, assumption, constant
   and number in the notes against it.
3. Write `papers/<key>/notes.md` (minute box → primer fold → divider → notes verbatim → source line).
4. Add the notes' and the primer's numbers to `scripts/check-math/<key>.py`; it must pass.
5. Audit the page on your own server (SKILL.md "Audit"): start `python scripts/serve.py <PORT>` once with Bash
   `run_in_background`, run `python scripts/sync.py` (local only), open `http://localhost:<PORT>/paper.html?id=<key>#notes`.
6. Append a section to `research/notes-rollout/<SESSION>-findings.md`:
   `## [n] <key> — <short title>` then: notes checked against (paper copy + pages), every fix made to the notes (original →
   fixed, with the paper's statement and page), anything you could not verify (and why), the primer's ideas in one line each.
   Write "No discrepancies found" when that is the result.

## Do not touch (shared files the coordinator merges)
`src/`, `index.html`, `paper.html`, `data/*.json` (you may regenerate `data/papers.json` locally with `scripts/sync.py`
for testing, but run `git checkout data/papers.json` before every commit), `research/findings.md`, `CLAUDE.md`, `.claude/`,
`scripts/*.py` other than your papers' check scripts, `scripts/check-math/common.py`, `scripts/check-math/results.txt`,
`public/`, other papers' folders, `papers/<key>/visual.js|math.md|meta.json`. If the renderer cannot express something
in a notes PDF, write it as close as the syntax allows, and describe the needed renderer change under
"Shared changes needed" at the top of your findings file instead of editing `src/`.

## Commit, do not push
Commit to your own branch after every 3–4 finished papers (message: `Notes pages <SESSION>: <keys>`, ending with the line
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Never push, merge or switch branches. At the end run each
of your check scripts once more and report: pages done, any not done and why, every fix made to the notes, anything
you could not verify, and any shared change needed. Drive is read-only and is not needed for this work.
