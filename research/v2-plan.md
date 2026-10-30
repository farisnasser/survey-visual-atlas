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
