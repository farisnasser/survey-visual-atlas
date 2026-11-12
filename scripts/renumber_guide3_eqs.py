"""Renumber every survey-equation reference in the v3 guide (guide/*/text.md, guide/*/viz.js,
scripts/check-math/guide3_numbers.py) after the survey's paragraph-by-paragraph rewrite (2026-11-12), using
research/survey3_renumber.json (built by scripts/survey3_renumber_map.py).

Handles "Eq. 5", "Eq. (5)", "Eqs. 6 and 7", "Eqs. (50)-(51)", "Eqs. 36-38", "Eqs. (8) and (70)", "Eq. (36), (37)".
A number whose equation the rewritten survey no longer contains becomes ⟦GONE n⟧ so it can be fixed by hand.
Run once; --dry prints the changes only."""
import os, re, sys, json
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
M = json.load(open(os.path.join(ROOT, "research", "survey3_renumber.json"), encoding="utf-8"))["eq"]
dry = "--dry" in sys.argv
NUM = r"\(?\d+\)?"
PAT = re.compile(r"(Eqs?\.\s?)(" + NUM + r"(?:(?:\s?[-–]\s?|\s(?:and|to)\s|,\s)" + NUM + r")*)")

def sub_num(tok):
    m = re.fullmatch(r"(\(?)(\d+)(\)?)", tok)
    new = M.get(m.group(2), "MISSING")
    if new == "MISSING":
        return tok + "⟦NOMAP⟧"
    if new is None:
        return m.group(1) + f"⟦GONE {m.group(2)}⟧" + m.group(3)
    return m.group(1) + str(new) + m.group(3)

def fix(match):
    body = re.sub(r"\(?\d+\)?", lambda t: sub_num(t.group(0)), match.group(2))
    return match.group(1) + body

files = [os.path.join(ROOT, "guide", d, f) for d in os.listdir(os.path.join(ROOT, "guide"))
         if os.path.isdir(os.path.join(ROOT, "guide", d)) and d != "compare" or d == "compare"
         for f in ("text.md", "viz.js") if os.path.exists(os.path.join(ROOT, "guide", d, f))]
files.append(os.path.join(ROOT, "scripts", "check-math", "guide3_numbers.py"))
tot = gone = 0
for p in files:
    s = open(p, encoding="utf-8").read()
    n2 = PAT.sub(fix, s)
    if n2 != s:
        ch = sum(1 for _ in PAT.finditer(s)); tot += ch; g = n2.count("⟦GONE"); gone += g
        print(f"{os.path.relpath(p, ROOT)}: {ch} refs, {g} gone")
        if not dry:
            open(p, "w", encoding="utf-8").write(n2)
print("total refs", tot, "gone", gone)
