"""Replace references to survey equations the rewritten survey no longer prints (⟦GONE n⟧ markers left by
renumber_guide3_eqs.py) with the survey section that now describes the idea in words. Generic patterns first,
then hand-written sentence fixes; prints whatever is left."""
import re, glob, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SEC = {12: "II-G", 16: "III-A", 17: "III-A", 21: "III-B", 22: "III-B", 33: "III-H", 35: "III-I", 39: "IV-A", 41: "IV-A",
       42: "IV-B", 43: "IV-B", 45: "IV-D", 46: "IV-D", 47: "IV-D", 49: "IV-E", 50: "IV-F", 51: "IV-F", 52: "IV-G",
       53: "IV-H", 54: "IV-H", 55: "IV-I", 57: "V-B", 58: "V-B", 60: "V-B", 61: "V-C", 62: "V-C", 64: "V-E", 65: "V-E",
       69: "V-G"}
G = r"\(?⟦GONE (\d+)⟧\)?"
# (survey Eq. X) / (survey Eqs. X and Y) / (Eq. X) / (Eqs. X-Y): the whole parenthetical becomes the section
P1 = re.compile(r"\((?:survey )?Eqs?\. " + G + r"(?:(?:,? and |\s?[-–]\s?)" + G + r")?\)")
HAND = {
    "guide/adaptive/viz.js": [("title: 'L2 inside Adam is not weight decay: Eq. (⟦GONE 33⟧) against Eq. (28)'",
                               "title: 'L2 inside Adam is not weight decay: the penalty inside the adaptive step against AdamW, Eq. (28)'")],
    "guide/kfac/text.md": [("gives the exact Fisher block of Eq. ⟦GONE 58⟧.", "gives the exact Fisher block written above."),
                           ("The text at the right repeats Eqs. (36) and ⟦GONE 60⟧ and", "The text at the right repeats Eq. (36), the inverse identity above, and")],
    "guide/kfac/viz.js": [("the exact Fisher block of Eq. ⟦GONE 58⟧ against", "the exact Fisher block (survey Section V-B) against")],
    "guide/lean/viz.js": [("// Eq. ⟦GONE 35⟧", "// survey Section III-I (in words)")],
    "guide/metric/viz.js": [("Eq. ⟦GONE 12⟧-(12) drawn literally", "Eq. (12) and its argmin form drawn literally")],
    "guide/momentum/text.md": [("The one-dimensional recurrence of Eq. ⟦GONE 21⟧", "The one-dimensional recurrence above"),
                               ("$\\max|r|$ of Eq. ⟦GONE 22⟧", "$\\max|r|$ of the characteristic polynomial above")],
    "guide/muon/text.md": [("the bound $\\sum_i\\sigma_i(G)$ of Eq. ⟦GONE 69⟧.", "the bound $\\sum_i\\sigma_i(G)$ above."),
                           ("no other update does: Eq. ⟦GONE 69⟧ checked numerically.", "no other update does: the bound above, checked numerically.")],
    "guide/muon/viz.js": [("// ---------- Eq. ⟦GONE 69⟧:", "// ---------- spectral-norm steepest descent (survey Section V-G, in words):"),
                          ("title: 'Eq. ⟦GONE 69⟧: under a spectral-norm budget", "title: 'Under a spectral-norm budget")],
    "guide/newton/text.md": [("This is Nesterov and Polyak's way of writing the survey's Eq. ⟦GONE 47⟧, with", "This is Nesterov and Polyak's way of writing the cubic model above, with")],
    "guide/quasinewton/text.md": [("BFGS with the inverse update of Eqs. ⟦GONE 50⟧ and ⟦GONE 51⟧,", "BFGS with the inverse update above,")],
    "guide/quasinewton/viz.js": [("// Eqs. ⟦GONE 50⟧-⟦GONE 51⟧", "// inverse BFGS update (survey Section IV-F, in words)"),
                                 ("'bfgs' (Eqs. 34-⟦GONE 51⟧ with a backtracking", "'bfgs' (Eq. 34 and the inverse update of Section IV-F, with a backtracking"),
                                 ("change y (Eqs. 34-⟦GONE 51⟧.", "change y (Eq. 34 and survey Section IV-F).")],
    "guide/fisher/text.md": [("| $\\nabla^2L$, Eq. ⟦GONE 53⟧ |", "| $\\nabla^2L$ |"),
                             ("| $H$ without the second term of Eq. ⟦GONE 53⟧ |", "| $H$ without the second term of the split above |"),
                             (", Eq. ⟦GONE 54⟧ | yes |", " | yes |")],
    "scripts/check-math/guide3_numbers.py": [("ch9: Eq. ⟦GONE 35⟧ reproduces", "ch9: Adafactor's factored V reproduces"),
                                             ("and Eq. ⟦GONE 69⟧", "and spectral-norm steepest descent"),
                                             ("ch16: Eq. ⟦GONE 69⟧: -UV^T", "ch16: spectral steepest descent: -UV^T")],
}
fs = glob.glob("guide/*/text.md") + glob.glob("guide/*/viz.js") + ["scripts/check-math/guide3_numbers.py"]
for f in fs:
    f = f.replace("\\", "/")
    s = open(f, encoding="utf-8").read(); s0 = s
    for a, b in HAND.get(f, []):
        if a not in s:
            a = re.sub(r"⟦GONE (\d+)⟧", r"(⟦GONE \1⟧)", a)
        if a not in s:
            print("not found (already fixed?):", f, a[:60]); continue
        s = s.replace(a, b)
    s = P1.sub(lambda m: f"(survey Section {SEC[int(m.group(1))]})", s)
    if s != s0:
        open(f, "w", encoding="utf-8").write(s)
left = [(f, l) for f in fs for l in open(f, encoding="utf-8").read().split("\n") if "⟦" in l]
for f, l in left:
    i = l.index("⟦"); print(f, "...", l[max(0, i - 100):i + 60])
print("left:", len(left))
