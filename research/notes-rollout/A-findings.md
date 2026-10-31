# Session A — in-page notes findings

## Shared changes needed
None so far.

Audit method used by this session: the in-app browser pane was hidden (its viewport reported 0 px), so each page was
loaded in a fixed 375 × 812 px iframe on `localhost:8571` and measured there (KaTeX errors, `__atlasErrors`, section
count, `scrollWidth − innerWidth`, elements past the right edge outside `.nt-eq`/`.nt-table`). Verbatim check: a
word-level diff (`difflib`) of the PDF text against `notes.md` with the primer, math and SVG removed; every remaining
difference is listed per paper (title block, box/fence keywords and math tokens only, unless a fix is logged).

## [1] murty1987some — Hardness in Quadratic and Nonlinear Programming
- Notes checked against: author scan public.websites.umich.edu/~murty/np.pdf, pp. 117–119 (abstract, Sec. 1, Sec. 2 Examples 1–2), read as images.
- Fixes to the notes: none. No discrepancies found. The notes make no theorem-numbered claim; every claim they do make
  (hardness of local-optimality / boundedness questions for indefinite QPs, worst-case scope) matches the abstract and
  Sec. 1 (p. 118). The "Verification limit" box describes how the notes were written and was kept verbatim; the paper
  itself was read for this page.
- Primary-source line: the notes' own "Source:" sentence (end of Sec. 4) moved into the `>` source line, verbatim.
- Primer (4 ideas): worst case is about a growing family (2^20 vs 2^60); the paper's Example 2 reduction of Subset Sum
  to a box QP, with d = (3, 5): d0 = 8 → min 0 at (1, 1), d0 = 4 → min 5/32 at (1, 3/16); copositivity vs eigenvalues
  ([[1,2],[2,1]] indefinite but copositive); the abstract's two NP-complete questions and the copositivity corollary.
  All numbers in `scripts/check-math/murty1987some.py`.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 4 sections, no overflow at 375 px.

## [2] blum1992training — Training a Tiny Network Can Be NP-Complete
- Notes checked against: author reprint people.csail.mit.edu/rivest/pubs/BR93.pdf (journal pp. 117–127): Sec. 4.1.2,
  Def. 1 and Fig. 1 (pp. 3–4: two hidden threshold nodes N1, N2 + output node N3), Theorem 2 (p. 5), Sec. 4.3.1–4.3.3
  (pp. 6–8), conclusions (p. 15–16: logistic units left open).
- Fixes to the notes: none. No discrepancies found. Remark (not changed): the notes write a threshold neuron as
  $\mathbf{1}\{w^\top x \ge b\}$ with outputs {0, 1}; the paper uses ±1 outputs and a strict inequality
  ($a_1x_1 + \dots + a_mx_m > a_0$). On a finite example set both conventions realise the same labelings, and the notes
  present it as their own definition, so it is not an error.
- Source line: the notes' "Sources:" paragraph, verbatim (DOI link parentheses URL-encoded so the link works).
- Primer (4 ideas): one threshold unit = one half-space (AND yes, XOR impossible — the w1 + w2 ≥ 2b > b contradiction);
  two hidden units compute XOR (h1 = 1{x1+x2 ≥ 1}, h2 = 1{x1+x2 ≥ 2}, ŷ = 1{h1 − h2 ≥ 1}); NP-completeness = short
  checkable certificate (O(n log n) bits per weight, the paper citing Raghavan) + reduction; the paper's Set-Splitting
  example with S1 = {s1, s3}, S2 = {s2}: a = (−1, 3, −1), b = (3, −1, 3). All checked in `blum1992training.py`.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 5 sections, no overflow at 375 px.

## [3] baldi1989neural — Neural Networks and Principal Component Analysis
- Notes checked against: Neural Networks 2:53–58 (open copy vision.jhu.edu/teaching/learning/deeplearning19/assets/Baldi_Hornik-89.pdf):
  abstract, Sec. 1 (AB = (AC)(C⁻¹B), p. 54), Sec. 2 main result and Facts 1–4 (pp. 54–55: unique local = global
  minimum at the top-p eigenvector projection, all other critical points saddles, rank-deficient ones included).
- Fixes to the notes: none. No discrepancies found. The notes' "nondegenerate critical configurations are saddles" is
  narrower than the paper (which also classifies rank-deficient critical points as saddles), but true.
- Silent artefact fixes: the PDF prints every section number twice ("1 1. Start with…", "3.1 3.1 The Hessian"); the page
  keeps one.
- Title: the notes' own title is the paper title ("Neural Networks and Principal Component Analysis"), used as-is.
  Box mapping follows the notes' colour legend: red "results" → theorem (key landscape conclusion), blue "why in ML" →
  key, amber "teaching clarification" → warn. The x → B → h → A → x̂ diagram is redrawn as SVG.
- Primer (4 ideas): spectral theorem on Σ = [[5,4],[4,5]] = 9u1u1ᵀ + 1u2u2ᵀ; uᵀΣu as a weighted average of eigenvalues
  (u = (1,0): 5); trace = total variance (10 = 9 + 1), the paper's eq. (10) error = total variance − kept eigenvalues;
  convex in each factor (Facts 1–2; (1 − 2b)² minimised at b = 1/2) but not jointly. All checked in `baldi1989neural.py`
  together with every number in the notes (gradient, H(0,0) eigenvalues ±2, the (3,2) projection, the trace identity,
  E(θ) = 1 + 8 sin²θ, AB = (AC)(C⁻¹B)).
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 8 sections, no overflow at 375 px.
