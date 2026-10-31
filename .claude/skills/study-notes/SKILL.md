---
name: study-notes
description: Turn one paper's notes PDF into the in-page study notes of the Survey Visual Atlas (papers/<Key>/notes.md, rendered by src/paper/notes.js): the one-minute summary, a first-principles primer, then the notes verbatim, checked against the original paper. Use when asked to put a paper's notes on its page, convert notes PDFs, or fix a notes.md.
---

# Notes PDF → in-page study notes

The reference page is **`papers/lee2016gradient/notes.md`** (live: `paper.html?id=lee2016gradient#notes`). Open it and
its notes PDF side by side before starting; every page must read like a sibling of it. When `papers/<key>/notes.md`
exists, `scripts/sync.py` sets `notes_page` and the paper page shows the article instead of the PDF viewer; the
header keeps a "Download PDF notes" button for `public/notes/<key>.pdf`.

## The order on every page (Faris's spec)
1. `# Title`: the notes' own topic title (e.g. "Why Gradient Descent Avoids Saddles"), not the paper title (the page
   header already shows the paper title, authors, venue, survey number, tags and links).
2. `:::minute The paper in one minute` — the notes' one-minute summary **verbatim**, keeping its own labels
   (Question / Model/Method / Finding / Why we care, or Problem / Approach / Main message, …), one bold label per
   paragraph. If the notes have no such summary, build it only from the notes' own sentences (opening paragraph,
   main result, limitation) and log that in your findings.
3. `:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the <N> ideas the paper stands on, built from scratch.`
   — the primer (below), closed by its own `:::`.
4. `:::divider Study notes` + `:::`
5. The notes **verbatim**: every section, heading and number, sentence, equation, table, list, box and figure, in
   the notes' order.
6. `> **Primary source:** …` line, verbatim from the notes when they have one; otherwise authors (year), *title*,
   venue, and the official link from `data/papers.json`.

Leave out only the PDF title block (the "Non-Convex Optimization — Study Notes" banner, REF / Paper numbers, tags,
tier, "Original paper" link text): the page header shows those.

## Verbatim means verbatim
- **Read the rendered pages, not only the text.** Text extraction breaks math (fractions, sub/superscripts, matrices,
  hats, norms). Render every page and look at it:
  `python -I -c "import fitz,sys; d=fitz.open(sys.argv[1]); [p.get_pixmap(dpi=150).save(f'{sys.argv[2]}/p{p.number+1}.png') for p in d]" public/notes/<key>.pdf <scratch dir>`
  and use `fitz` `get_text()` only to copy the wording. Transcribe each equation symbol for symbol in KaTeX.
- Same words, same headings, same section numbers, same bold/italic emphasis. Do not reword, shorten, merge or
  reorder. Do not "improve" the explanation.
- Box mapping: the notes' key-idea / intuition boxes → `:::key Title`; boxed theorem/proposition statements →
  `:::theorem Title`; warnings, "does not prove", common confusions → `:::warn Title`; other neutral boxes (e.g. the
  "Papers Master — suggested 'Our notes'" box, copy texts) → `:::plain Title`. Box titles verbatim. Two-column layouts →
  `:::cols` / `:::col`. Boxed equations stay `\boxed{}`; underbraces stay `\underbrace{}`.
- **Figures and diagrams:** redraw each one as inline SVG in `:::figure <caption verbatim>` with the same content
  (curves, arrows, labels, points, axis names) and the Atlas colour language: axes `#8e99bd`, text `#e8ecf8`,
  positive curvature / stable blue `#7cc4ff`, negative curvature / unstable red `#ff5d73`, flat amber `#ffc857`,
  minima green `#5fe0b0`, optimizer coral `#ff7a59` / violet `#b28cff`; `font-size="9"`; a `viewBox` wide enough that
  no label clips; `role="img"` + an `aria-label` describing the figure; marker ids prefixed `nt`. A plotted curve must
  be the real function the notes name (compute its points; do not sketch by eye). Lee's §4 figure is the model.
- **Change something only if it is wrong.** Faris: "IF and ONLY IF you understand the paper better than the notes
  and see a discrepancy" — a wrong theorem/section/page number, constant, exponent, assumption, rate, or a claim the
  paper does not make. Then make the smallest edit that fixes it, keep the notes' wording around it, and log the
  original text, the fix and the paper's exact statement (with page/theorem number) in your findings. Clear
  PDF-generation artefacts (a broken glyph, a duplicated word) may be fixed silently. Style is never a reason.

## Checking (no inaccuracies, none)
- **Read the original paper** (official PDF from `data/papers.json` `pdf`/`link`; arXiv when the venue copy is paywalled).
  Save downloads in their own empty scratch folder and read them with `python -I`. Check every theorem / lemma /
  proposition / equation / figure / section / page number the notes cite, every assumption, constant and rate, and
  every claim about what the paper shows, against the paper itself.
- **Every number** in the notes and the primer (worked examples, iterates, eigenvalues, rates, table values that are
  computed rather than reported) is reproduced in `scripts/check-math/<key>.py` under a comment
  `# in-page notes (papers/<key>/notes.md): …`, with `check(...)` lines, and the script still ends with `done()` and
  passes. Numbers the notes quote from the paper's experiments are checked against the paper's tables/text instead
  (say so in the findings). See `scripts/check-math/lee2016gradient.py` for the pattern.
- Facts in the primer that are standard (definitions, textbook identities) must be stated correctly and in the form
  the paper uses (same symbols as the notes: if the notes write $\eta$ for the step size, so does the primer).

## The first-principles primer
For a master's-level ML student who knows calculus, linear algebra and basic probability, not this paper.
- **3–6 numbered ideas** as `### 1. <short claim-like heading>`, each 1–3 short paragraphs, at most one display
  equation, then `:::key Putting the <N> ideas together` (or "… the ideas together") tying them to the paper's main
  result in 2–4 sentences, ending by pointing to the notes ("made precise in the notes below").
- Build each idea from something the student already knows (the update rule, a Taylor expansion, a 2×2 example, a
  coin flip) up to the concept the paper needs. Concrete beats abstract: one small worked example with real numbers
  (checked in the script) is worth a paragraph of definitions.
- **Tone: learn it from the notes.** Short declarative sentences. Present tense. Bold the term at the point it is
  defined. Say what a symbol means the first time it appears. Plain words over jargon; no hype, no "simply", no
  "it is easy to see", no rhetorical questions stacked up, no emoji. Separate what the paper proves from intuition.
- Do not duplicate the notes. If the notes already open with a first-principles section, the primer covers the
  prerequisites those sections assume (e.g. what NP-hardness means, what a Lipschitz gradient buys, what "with high
  probability" means), not the same derivation again.
- Never claim anything about the paper that the paper does not state. When the primer simplifies (a quadratic
  model of a nonconvex loss, a 2-D picture of a d-dimensional fact), say so in one clause.

## notes.md syntax (src/paper/notes.js)
`##` = numbered section (gets a contents entry; write the notes' numbering into the heading: `## 3. …`) · `###` sub-head ·
`$$…$$` display (may span lines) · `$…$` inline (must stay on one line; no `$` signs as currency) · `**b**` `*i*` `` `c` `` `[t](url)` ·
lists `- ` / `1. ` with continuation lines (including `$$` blocks) indented 3 spaces · pipe tables (header, `|---|:-:|` row) ·
`:::kind Title … :::` (minute, key, theorem, warn, ours, plain) · `:::cols`/`:::col` · `:::figure Caption` (raw SVG) ·
`:::divider Label` · `:::fold Title | subtitle` · `> source line`. Fences nest; every opener needs its own `:::`.
**Inside table cells never write a bare `|`** (it splits the cell): use `\lVert x\rVert`, `\lvert x\rvert`, `\mid`.
KaTeX 0.16.9 only (no `\bm`, use `\boldsymbol`; `\operatorname{}` for named operators).

## Audit before you call a paper done
1. `python scripts/check-math/<key>.py` passes.
2. `python scripts/sync.py` (local), open `http://localhost:<PORT>/paper.html?id=<key>#notes` and run in the page:
   `[document.querySelectorAll('#notes .katex-error').length, window.__atlasErrors, document.querySelectorAll('#notes .nt-h').length]`
   → `0`, `[]`, the notes' section count. Open the primer fold, click two contents links.
3. At 375 px width (`resize_window` mobile): `document.documentElement.scrollWidth - innerWidth` is `0`, and no
   equation panel or table forces sideways page scroll (`.nt-eq` may scroll inside itself for very long equations).
4. Compare the rendered article against the PDF pages one last time, section by section: nothing missing, nothing added
   outside the primer and logged fixes.
