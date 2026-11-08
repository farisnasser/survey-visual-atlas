"""One-off: a new chapter 6 (survey II-G/H/I) is inserted; chapters 6..15 become 7..16.
Renumbers 'chapter N', 'chapters N to M', 'Chapters 7, 9 and 13', 'chapters 6–8' in guide/*/text.md and guide.json."""
import json, os, re
ROOT = os.path.join(os.path.dirname(__file__), "..", "guide")
f = lambda n: n + 1 if n >= 6 else n
pat = re.compile(r"(?i)\b(chapters?\s+)(\d+(?:\s*(?:–|-|to|and|,)\s*\d+)*)")
def fix(m):
    return m.group(1) + re.sub(r"\d+", lambda k: str(f(int(k.group()))), m.group(2))
total = 0
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d, "text.md")
    if not os.path.isfile(p):
        continue
    s = open(p, encoding="utf-8").read()
    s2, n = pat.subn(fix, s)
    if s2 != s:
        open(p, "w", encoding="utf-8").write(s2)
    total += n
    print(d, n)
g = json.load(open(os.path.join(ROOT, "guide.json"), encoding="utf-8"))
for c in g["chapters"]:
    c["n"] = f(c["n"])
json.dump(g, open(os.path.join(ROOT, "guide.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("references rewritten:", total)
