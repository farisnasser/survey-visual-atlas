"""Guide: every [[term]] in guide/*/text.md resolves to course/glossary.json or guide/glossary.json, every guide term's
`sec` anchor exists in its chapter, and every :::viz slot has a visual name and a following :::read box."""
import json, os, re
from common import check, done, ROOT

print("guide/terms")
course = json.load(open(os.path.join(ROOT, "course", "glossary.json"), encoding="utf-8"))
gpath = os.path.join(ROOT, "guide", "glossary.json")
guide = json.load(open(gpath, encoding="utf-8"))
G = json.load(open(os.path.join(ROOT, "guide", "guide.json"), encoding="utf-8"))
slug = lambda h: "n-" + re.sub(r"[^a-z0-9]+", "-", h.lower()).strip("-")
anchors, texts = {}, {}
for c in G["chapters"]:
    t = open(os.path.join(ROOT, "guide", c["id"], "text.md"), encoding="utf-8").read()
    texts[c["id"]] = t
    anchors[c["id"]] = {slug(h) for h in re.findall(r"^#{2,3} (.+)$", t, re.M)}
used = set()
for cid, t in texts.items():
    for m in re.finditer(r"\[\[([\w-]+)(?:\|[^\]]+)?\]\]", t):
        used.add(m.group(1))
missing = sorted(u for u in used if u not in course and u not in guide)
check("every [[term]] resolves", not missing, f"missing: {missing}")
chip = sorted({t for c in G["chapters"] for t in c.get("terms", []) if t not in course and t not in guide})
check("every map chip term resolves", not chip, f"missing: {chip}")
bad = [(k, v.get("sec")) for k, v in guide.items() if v["ch"] in anchors and v.get("sec") and v["sec"] not in anchors[v["ch"]]]
check("every guide-glossary anchor exists in its chapter", not bad, f"{bad}")
nochap = [k for k, v in guide.items() if v["ch"] not in anchors]
check("every guide-glossary term points at a guide chapter", not nochap, f"{nochap}")
# every visual is followed by a How-to-read box
for cid, t in texts.items():
    blocks = re.split(r"^:::viz ", t, flags=re.M)[1:]
    lacking = [b.split("\n")[0] for b in blocks if not re.search(r"^:::read ", b.split(":::viz")[0], re.M)]
    check(f"{cid}: every :::viz is followed by a :::read box", not lacking, f"{lacking}")
    surv = re.split(r"^:::survey ", t, flags=re.M)[1:]
    lacking = [b.split("\n")[0][:40] for b in surv if not re.search(r"^:::read ", b, re.M)]
    check(f"{cid}: every survey figure is followed by a :::read box", not lacking, f"{lacking}")
print(f"   {len(used)} distinct terms used; guide glossary has {len(guide)} entries")
done()
