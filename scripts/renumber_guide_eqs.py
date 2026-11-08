"""One-off: renumber survey-equation references in guide/*/text.md after the survey's first-principles pass
(2026-11-08). Paper-internal equation references (Cohen et al.'s Eq. (1), Damian et al.'s Eq. (3)/(4),
Kalra's Eqs. 1-2, Lewkowycz's Eqs. 6-7) are left alone."""
import os, sys
ROOT = os.path.join(os.path.dirname(__file__), "..", "guide")
EDITS = {
    "multiplier": [("survey Eq. (6)", "survey Eq. (9)"), ("survey Eq. (7)", "survey Eq. (10)"), ("survey Eq. (8)", "survey Eq. (11)")],
    "stages": [("survey Eq. (10)", "survey Eq. (16)"), ("the exact solution of Eq. (10)", "the exact solution of survey Eq. (16)"),
               ("following Saxe et al.'s Eq. (10)", "following the sigmoidal law of survey Eq. (16)")],
    "precursors": [("survey Eq. (11)", "survey Eq. (17)"), ("of Eq. (11)", "of Eq. (17)")],
    "why": [("survey Eq. (12)", "survey Eq. (18)")],
    "flow": [("survey Eq. 12", "survey Eq. 18"), ("survey Eq. (13)", "survey Eq. (19)")],
    "beyond": [("survey Eq. (14)", "survey Eq. (20)"), ("batch sharpness, Eq. (14)", "batch sharpness, Eq. (20)"), ("condition (11)", "condition (17)")],
    "compare": [("survey Eq. 12", "survey Eq. 18"), ("(Eq. 11)", "(Eq. 17)"), ("of Eq. (12)", "of Eq. (18)"), ("Under Eq. (14)", "Under Eq. (20)"), ("condition (11)", "condition (17)")],
    "open": [("of Eq. (14)", "of Eq. (20)"), ("of Eq. (12)", "of Eq. (18)")],
}
bad = 0
for ch, reps in EDITS.items():
    p = os.path.join(ROOT, ch, "text.md")
    s = open(p, encoding="utf-8").read()
    for a, b in reps:
        n = s.count(a)
        if n == 0:
            print(f"{ch}: NOT FOUND {a!r}"); bad += 1
        s = s.replace(a, b)
        print(f"{ch}: {a!r} -> {b!r} x{n}")
    open(p, "w", encoding="utf-8").write(s)
sys.exit(1 if bad else 0)
