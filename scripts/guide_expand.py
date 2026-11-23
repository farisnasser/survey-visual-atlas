"""Rewrite helper for the v3 guide (2026-11-23 rewrite from the study notes).

A chapter is written as guide/<id>/text.src.md. Placeholders pull blocks, unchanged, from the guide as it was at the
commit given by BASE (the last aligned version):
  @@viz NAME@@            the ':::viz NAME' slot and the ':::read' box that follows it (searched in every chapter)
  @@read NAME@@           only the slot, with a new read box written right after it in the source
  @@survey N@@            the ':::survey Fig. N' block and its ':::read' box
  @@survey N->M@@         the same block, renumbered from Fig. N to Fig. M (image path too)
Writes guide/<id>/text.md, then lints every chapter for references that do not exist in the submitted survey
(appendices, equations above 40, Figs. above 10, Tables above V).
usage: python scripts/guide_expand.py [chapter ...]"""
import os, re, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = "2458018"
CHAPTERS = [d for d in sorted(os.listdir(os.path.join(ROOT, "guide"))) if os.path.isdir(os.path.join(ROOT, "guide", d))]

def old(ch):
    try:
        return subprocess.run(["git", "show", f"{BASE}:guide/{ch}/text.md"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
    except subprocess.CalledProcessError:
        return ""
OLD = {ch: old(ch) for ch in CHAPTERS}

def fence(lines, i):
    """lines[i] opens a ::: block; return index after its closing ':::'"""
    depth = 1; i += 1
    while i < len(lines):
        l = lines[i].strip()
        if re.match(r"^:::\S", l): depth += 1
        elif l == ":::":
            depth -= 1
            if depth == 0: return i + 1
        i += 1
    return i

def grab(kind, key, with_read=True):
    for ch, txt in OLD.items():
        L = txt.split("\n")
        for i, l in enumerate(L):
            ok = (kind == "viz" and l.strip() == f":::viz {key}") or (kind == "survey" and l.startswith(f":::survey Fig. {key} "))
            if not ok: continue
            j = fence(L, i); block = L[i:j]
            if with_read:
                k = j
                while k < len(L) and not L[k].strip(): k += 1
                if k < len(L) and L[k].startswith(":::read"):
                    e = fence(L, k); block += [""] + L[k:e]
            return "\n".join(block)
    raise KeyError(f"{kind} {key} not found in the base guide")

def expand(src):
    def rep(m):
        kind, arg = m.group(1), m.group(2).strip()
        if kind == "viz": return grab("viz", arg)
        if kind == "read": return grab("viz", arg, with_read=False)
        a, _, b = arg.partition("->")
        blk = grab("survey", a.strip())
        if b:
            b = b.strip()
            blk = blk.replace(f"Fig. {a.strip()} of the survey", f"Fig. {b} of the survey").replace(f"fig{a.strip()}.webp", f"fig{b}.webp")
        return blk
    out = re.sub(r"@@(viz|read|survey)\s+([^@]+)@@", rep, src)
    for a, b in FIXUPS:
        out = out.replace(a, b)
    return out

# stale fragments inside reused blocks (the submitted survey has no appendix: equations stop at (40), figures at 10)
FIXUPS = [
    ("Rotated by $45^\\circ$, the matrix of Eq. (41).", "Rotated by $45^\\circ$, the matrix $H_{\\text{rot}}$ above."),
    ("the curved valley $L = (1 - x)^2 + 10(y - x^2)^2$ of the survey's Fig. 10", "the curved valley $L = (1 - x)^2 + 10(y - x^2)^2$"),
    ("The curved valley $L = (1 - x)^2 + 10(y - x^2)^2$ of the survey's Fig. 10 in 3D", "The curved valley $L = (1 - x)^2 + 10(y - x^2)^2$, a Rosenbrock-type test function, in 3D"),
    ("The same measured runs as Fig. 9,", "The same measured runs as Fig. 10,"),
    ("The right-hand side of Eq. (47) as a function", "The right-hand side of the $T$-step bound above as a function"),
    ("(printed under the panels, and in full in the survey's Table VIII)", "(printed under the panels)"),
]

BAD = [(r"\bAppendi(x|ces)\b", "appendix reference"), (r"\bEqs?\.?\s*\(?(4[1-9]|[5-9]\d)\)?", "equation > 40"),
       (r"\bFig(s|ure)?\.?\s*1[1-9]\b", "figure > 10"), (r"\bTable (VI{1,3}|IX)\b", "table > V"), (r"\bII-H\b|\bVII-E\b", "removed section")]

def lint():
    n = 0
    for ch in CHAPTERS:
        p = os.path.join(ROOT, "guide", ch, "text.md")
        if not os.path.exists(p): continue
        for k, line in enumerate(open(p, encoding="utf-8").read().split("\n"), 1):
            for pat, why in BAD:
                if re.search(pat, line):
                    n += 1; print(f"  lint {ch}:{k} {why}: {line[:110]}")
    print("lint issues:", n)

if __name__ == "__main__":
    todo = sys.argv[1:] or [c for c in CHAPTERS if os.path.exists(os.path.join(ROOT, "guide", c, "text.src.md"))]
    for ch in todo:
        src = open(os.path.join(ROOT, "guide", ch, "text.src.md"), encoding="utf-8").read()
        open(os.path.join(ROOT, "guide", ch, "text.md"), "w", encoding="utf-8").write(expand(src))
        print("built", ch)
    lint()
