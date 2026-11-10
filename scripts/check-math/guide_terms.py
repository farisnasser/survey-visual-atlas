"""Guide: every [[term]] in <guide>/*/text.md resolves to course/glossary.json or <guide>/glossary.json, every guide
term's `sec` anchor exists in its chapter, and every :::viz slot / :::survey figure is followed by a :::read box.
Checks both guides: guide/ (survey v3, the home page) and guide-v2/ (survey v2, guide.html?v=2)."""
import json, os, re, sys
from common import check, done, ROOT

course = json.load(open(os.path.join(ROOT, "course", "glossary.json"), encoding="utf-8"))
slug = lambda h: "n-" + re.sub(r"[^a-z0-9]+", "-", h.lower()).strip("-")
for GUIDE in (sys.argv[1:] or ["guide", "guide-v2"]):
    print(f"{GUIDE}/terms")
    guide = json.load(open(os.path.join(ROOT, GUIDE, "glossary.json"), encoding="utf-8"))
    G = json.load(open(os.path.join(ROOT, GUIDE, "guide.json"), encoding="utf-8"))
    anchors, texts = {}, {}
    for c in G["chapters"]:
        if not c.get("ready"):
            continue
        t = open(os.path.join(ROOT, GUIDE, c["id"], "text.md"), encoding="utf-8").read()
        texts[c["id"]] = t
        anchors[c["id"]] = {slug(h) for h in re.findall(r"^#{2,3} (.+)$", t, re.M)}
    used = set()
    for cid, t in texts.items():
        for m in re.finditer(r"\[\[([\w-]+)(?:\|[^\]]+)?\]\]", t):
            used.add(m.group(1))
    missing = sorted(u for u in used if u not in course and u not in guide)
    check(f"{GUIDE}: every [[term]] resolves", not missing, f"missing: {missing}")
    chip = sorted({t for c in G["chapters"] for t in c.get("terms", []) if t not in course and t not in guide})
    check(f"{GUIDE}: every map chip term resolves", not chip, f"missing: {chip}")
    bad = [(k, v.get("sec")) for k, v in guide.items() if v["ch"] in anchors and v.get("sec") and v["sec"] not in anchors[v["ch"]]]
    check(f"{GUIDE}: every guide-glossary anchor exists in its chapter", not bad, f"{bad}")
    nochap = [k for k, v in guide.items() if v["ch"] not in anchors]
    check(f"{GUIDE}: every guide-glossary term points at a guide chapter", not nochap, f"{nochap}")
    for cid, t in texts.items():
        blocks = re.split(r"^:::viz ", t, flags=re.M)[1:]
        lacking = [b.split("\n")[0] for b in blocks if not re.search(r"^:::read ", b.split(":::viz")[0], re.M)]
        check(f"{GUIDE}/{cid}: every :::viz is followed by a :::read box", not lacking, f"{lacking}")
        surv = re.split(r"^:::survey ", t, flags=re.M)[1:]
        lacking = [b.split("\n")[0][:40] for b in surv if not re.search(r"^:::read ", b, re.M)]
        check(f"{GUIDE}/{cid}: every survey figure is followed by a :::read box", not lacking, f"{lacking}")
    print(f"   {len(used)} distinct terms used; {GUIDE} glossary has {len(guide)} entries")
done()
