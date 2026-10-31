# Round 2 (2026-10-30): survey references + 19 new notes

Source of truth for the paper list is now the final survey PDF (`NonConvex_Survey_Final.pdf`, 26 pages, 111 references).
Organization = the survey's Fig. 5 taxonomy (Sections III–VII leaves); references outside the taxonomy go under
"Background" (Sec. I–II) with the sections that cite them. Notes filenames use an older working numbering ("Ref71" etc.),
NOT the survey numbering — always map by author + title.

Scratch: `AppData/Local/Temp/claude/C--Users-Asus-faris-brain/64367bc6-41e9-4a4e-85a6-05a2eb640d6d/scratchpad/v2/`
(survey.pdf, survey_fitz.txt, refs.json, cites.json, notes/, papers/, txt/).

| Survey # | Key | Notes file | Paper source | Status |
|---|---|---|---|---|
| 2 | murty1987some | Ref_02_Murty_Kabadi_1987_Detailed_Notes.pdf | paywalled; verified via Ahmadi & Zhang arXiv 2008.12170 | read + verified |
| 3 | blum1992training | Ref_03_Blum_Rivest_1992_Detailed_Notes.pdf | NeurIPS 1988 version (papers.neurips.cc …3def184a…) | read + verified |
| 10 | jin2018accelerated | Ref_10_Jin_Netrapalli_Jordan_2018_Detailed_Notes (1).pdf | arXiv 1711.10456 | read + verified |
| 11 | fang2018spider | Ref_11_Fang_SPIDER_2018_Detailed_Notes.pdf | arXiv 1807.01695 | read + verified |
| 12 | arjevani2020second | Ref_12_Arjevani_etal_2020_Detailed_Notes.pdf | arXiv 2006.13476 | read + verified |
| 33 | allenzhu2018neon2 | Ref77_Allen_Zhu_Li_2018.pdf | arXiv 1711.06673 | read + verified |
| 53 | nguyen2017loss | Nguyen_Hein_2017_Study_Notes.pdf | PMLR v70 nguyen17a | read + verified |
| 58 | bhojanapalli2016global | Bhojanapalli_2016_Study_Notes.pdf | arXiv 1605.07221 | read + verified |
| 60 | ge2017no | Ge_Jin_Zheng_2017_Study_Notes.pdf | arXiv 1704.00708 | read + verified |
| 69 | daneshmand2018escaping | Daneshmand_2018_Study_Notes.pdf | PMLR v80 daneshmand18a | read + verified |
| 70 | fang2019sharp | 41_fang2019sharp.pdf | arXiv 1902.00247 | read + verified |
| 71 | gurbuzbalaban2021heavy | 42_gurbuzbalaban2021heavy.pdf | arXiv 2006.04740 | read + verified |
| 72 | azizian2025global | 43_azizian2025global.pdf | arXiv 2503.16398 | read + verified |
| 74 | tripuraneni2018stochastic | 44_tripuraneni2018stochastic.pdf | arXiv 1711.02838 | read + verified |
| 75 | carmon2018analysis | 45_carmon2018analysis.pdf | arXiv 1806.09222 | read + verified |
| 76 | carmon2019gradient | Ref71_Carmon_Duchi_2019.pdf | arXiv 1612.00547 | read + verified |
| 77 | wang2020cubic | Ref72_Wang_2020.pdf | PMLR v115 wang20c | read + verified |
| 79 | reddi2018generic | Ref74_Reddi_2018.pdf | PMLR v84 reddi18a | read + verified |
| 81 | xu2018first | Ref76_Xu_Jin_Yang_2018.pdf | arXiv 1711.01944 | read + verified |

Existing 16 visuals map to survey refs: baldi1989neural [4], dauphin2014identifying [5], lee2016gradient [7], jin2017escape [8],
du2019gradient [14], allenzhu2019convergence [15], zhang2026saddle [18], saxe2014exact [30], kawaguchi2016deep [31],
kalra2025universal [37], galli2026flatland [39], ghadimi2013stochastic [47], laurent2018deep [52], safran2018spurious [55],
wu2025loss [56], li2018visualizing [62].

## Survey statements to double-check against the originals
- [3] survey: "training a network with only three hidden nodes is NP-complete". Blum & Rivest's 3-node net = 2 hidden + 1 output.
- [2] survey: "deciding whether a point is even a local minimum of a general smooth function is NP-hard" — OK (M&K: NP-complete for indefinite QP / copositivity; quartic local-min testing NP-hard per Ahmadi–Zhang).
- [72] paper prints p1 ≈ (−2.573, 1.029) but its own formula's minimizer is (−2.533, 1.013).
- [72] notes flag: (D.44) called a martingale is a discounted sum (E[M_{n+1}|F_n] = (1−γμ)M_n) — verified.
- [70] notes' "Theorem 5" is Theorem 1 in arXiv 1902.00247.
- [58] arXiv v2 says (2r)-RIP; the NeurIPS published version says (4r, 1/10) — we follow the published version.
- Fig. 5 caption: "The four branches correspond to Sections III–VII" — the figure has five branches (III, IV, V, VI, VII).
- Fig. 5 caption: "a work … appears under its primary contribution" — yet [5], [6], [57] and [66] each appear in two leaves. The hub follows the figure as drawn.
- Every reference is cited in the text: [26] only inside the range [25]–[27] (Sec. I-B).

# Round 3 (2026-10-30/31): 64 pages built by three parallel sessions

Notes for 64 more references were added to Drive. They were mapped to survey numbers, then built in three git worktrees
(branches round3-A/B/C, brief in `research/round3/BRIEF.md`), merged into main, synced, checked and audited.
Per-paper findings are in `research/findings.md` (originals in `research/round3/*-findings.md`), and links were merged
into `data/atlas_refs.json`. Result: 111 references, 98 with notes, 99 visuals (Kawaguchi is visual-only).

Still without notes (13): [1], [6], [9], [13], [28], [29], [31], [38], [73], [78], [80], [83], [111]. [31] Kawaguchi has a visual.

## Round 3 survey statements to double-check
- [98] wrong author order: the survey lists "Z. Wang, Z. Li, and J. Li". The paper's authors are Zhouzi Li, Zixuan Wang and Jian Li.
- [54] "if the network is large enough …": Theorem 1 has no size condition. Size enters only in Theorem 2 (r > card(X)), and the λ > 0 regularizer is essential.
- [57] "convergence guarantees" holds only up to an O(βσ²_int) neighbourhood, which is vacuous for the paper's Example 2.
- [64] the arXiv v3 (2025) correction needs an invertible Hessian at the saddle, so non-isolated saddles are not covered. Consider citing v3.
- [65] "Robbins–Monro step sizes": the paper goes beyond them (γ_n ∝ 1/n^p).
- [66] the analysis is of deterministic AdaGrad (full gradients).
- [84] SpiderBoost is cited in Sec. V (escape algorithms), but the paper proves only first-order stationarity.
- [92] Nguyen 2021 is an existence/width result. The guaranteed step for a small net is about 1.7e-16, so it should not be described as practical.
- [36] "roughly two orders of magnitude": √(1 − β2) ≈ 1/32 is about 1.5 orders. This repeats the paper's own rounding.

## Round 3 notes errors (content otherwise correct)
- Most notes headers carry old working reference numbers.
- jin2021nonconvex: §3 describes the 2017 perturbation rule, not the JACM Algorithm 1.
- panageas2019first: missing the v3 invertible-Hessian caveat.
- haeffele2017global: the cited supplement URL is dead. The result is Prop. 19 of arXiv 1506.07540.
- mei2019mean: says "Theorem 3", but it is Theorem 1.
- li2020learning: says "Theorem 2", but it is Theorem 1.2.
- agarwala2023sam: Theorems 2.2 and 2.3 are swapped.
- polyak1964some: drops α in one remark (trivial).

# In-page notes rollout (2026-10-31): notes PDFs -> papers/<key>/notes.md for all 98 papers

Lee et al. 2016 was the approved test page. The other 97 were built by six parallel sessions (branches notes-A..F,
brief `research/notes-rollout/BRIEF.md`, spec `.claude/skills/study-notes/SKILL.md`), merged into main and audited
(98/98 pages: no KaTeX or runtime errors, no overflow at 375 px or 1280 px). Per-paper verification, including what
each primer covers, is in `research/notes-rollout/{A..F}-findings.md`.

## Changes made to the notes (each logged with the paper's wording and page in the findings files)
- jin2021nonconvex §3: main PGD (Algorithm 1) adds Gaussian noise every step, not the 2017 occasional kick.
- sun2020optimization §2: "if η ≤ 1/ℓ" → "if η = 1/ℓ" (the stated bound follows only there); dead source URL replaced.
- chizat2018global: "the flow converges to global minimizers" → "if the flow converges, the limit is a global minimizer".
- bhojanapalli2016global §8: the paper's Sec. 5 Examples 1–2 are false as printed; the notes now say so.
- panageas2019first §3: qualified with the authors' v3 correction (invertible Hessian at the saddle).
- wang2020cubic §3–4: inexact-Hessian assumption (Assumption 3) and solver-accuracy conditions made explicit.
- zhang2019lookahead "Section 4" → "Section 5"; wang2019spiderboost reading-guide sections; mei2019mean "Theorem 3" →
  "Theorem 1"; agarwala2023sam swapped theorem numbers; gurbuzbalaban2021heavy §3 heading; li2020learning §2 worked
  number (Q = 0.005).
- Cross-references: old draft "Ref NN" numbers inside the notes now use survey numbers (matched by the named authors),
  e.g. NEON2 → Ref 33, SSRGD → Ref 87, SpiderBoost → Ref 84. Title-block metadata (tier, tags, old numbers) is dropped.

## Survey statements to double-check (new)
- [58] Bhojanapalli, Neyshabur & Srebro: do not cite the paper's Sec. 5 examples as proof that spurious minima exist without RIP.
- fang2019sharp and tripuraneni2018stochastic notes follow the publisher numbering (COLT / NeurIPS), not the arXiv copy linked in the page header.

## Renderer change from the rollout
- Header-less tables (many notes' "Quick reference") are written with an all-empty header row; `src/paper/notes.js`
  now renders those without a header (they used to show a thin empty strip on desktop).
