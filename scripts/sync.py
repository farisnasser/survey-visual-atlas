"""Sync the Atlas: survey references -> data/papers.json, Drive notes PDFs -> public/notes/<Key>.pdf.

Sources (all read-only):
  data/survey.json        built from the final survey PDF by scripts/build_survey.py — every reference, where it is cited,
                          and the Fig. 5 taxonomy. This is the source of truth for which papers the Atlas lists.
  data/atlas_refs.json    survey reference number -> Atlas key + official paper links (hand-maintained)
  data/notes_map.json     Atlas key -> notes file name in the Drive notes folder + Drive id (hand-maintained)
  data/legacy_sheet.json  tracker summaries from the old Papers Master sheet (frozen; shown as "AI-drafted")
  papers/<Key>/meta.json  per-paper result, tags, tabs (written with each visual)

usage:
  python scripts/sync.py [--notes-dir <folder with the downloaded notes PDFs>]

Notes files are matched by the exact file name recorded in notes_map.json (the Drive names follow no single
pattern, e.g. "Ref71_Carmon_Duchi_2019.pdf" uses an old working number). PDFs in --notes-dir that are not in the
map are listed so they can be mapped by hand after reading their title page; nothing is guessed.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import unicodedata
from urllib.parse import quote_plus

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STOP = {"a", "an", "the", "on", "in", "of", "for", "to", "how", "why", "what", "when", "does", "do", "is", "are", "can", "and", "with", "via", "from", "by"}


def load(name, default=None):
    p = os.path.join(ROOT, "data", name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default


def author_list(s):
    s = re.sub(r",?\s+and\s+", ", ", s)
    return [a.strip() for a in s.split(",") if a.strip() and a.strip() != "et al."]


PARTICLES = {"von", "van", "de", "da", "der", "den", "di", "le"}


def surname(a):
    w = re.sub(r"\bet al\.?", "", a).split()
    if not w:
        return ""
    k = len(w) - 1
    while k > 0 and w[k - 1].lower() in PARTICLES:
        k -= 1
    return " ".join(w[k:])


def ascii_key(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())


def short_name(authors, year):
    al = author_list(authors)
    sn = [surname(a) for a in al]
    y = f" {year}" if year else ""
    if len(sn) == 1:
        return sn[0] + y
    if len(sn) == 2:
        return f"{sn[0]} & {sn[1]}{y}"
    return f"{sn[0]} et al.{y}" if sn else (authors[:30] + y)


def auto_key(r, taken):
    sn = ascii_key(surname(author_list(r["authors"])[0])) if r["authors"] else "anon"
    word = next((w for w in re.findall(r"[A-Za-z]+", r["title"]) if w.lower() not in STOP), "x").lower()
    k = f"{sn}{r['year'] or ''}{word}"
    while k in taken:
        k += "b"
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes-dir")
    a = ap.parse_args()
    S = load("survey.json")
    amap = load("atlas_refs.json")["refs"]
    nmap = load("notes_map.json", {})
    legacy = load("legacy_sheet.json", {"refs": {}})["refs"]
    by_file = {v["file"]: k for k, v in nmap.items()}
    if a.notes_dir:
        os.makedirs(os.path.join(ROOT, "public", "notes"), exist_ok=True)
        for f in sorted(os.listdir(a.notes_dir)):
            if not f.lower().endswith(".pdf"):
                continue
            k = by_file.get(f)
            if k:
                shutil.copyfile(os.path.join(a.notes_dir, f), os.path.join(ROOT, "public", "notes", f"{k}.pdf"))
                print(f"notes  {f:58s} -> {k}")
            else:
                print(f"UNMAPPED notes file {f} — read its title page, then add it to data/notes_map.json")
    leaves = {}
    for br in S["taxonomy"]:
        for lf in br["leaves"]:
            for n in lf["refs"]:
                leaves.setdefault(n, []).append([br["id"], lf["name"]])
    taken = {v["key"] for v in amap.values()}
    papers = []
    for n_s, r in S["refs"].items():
        n = int(n_s)
        m = amap.get(n_s)
        key = m["key"] if m else auto_key(r, taken)
        taken.add(key)
        meta_p = os.path.join(ROOT, "papers", key, "meta.json")
        meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
        lg = legacy.get(n_s, {})
        cited = []
        for sec, _ in r["cited_in"]:
            if sec not in cited:
                cited.append(sec)
        papers.append({
            "n": n, "key": key, "title": r["title"] or r["text"], "authors": r["authors"], "year": r["year"], "venue": r["venue"],
            "short": short_name(r["authors"], r["year"]), "ref": r["text"],
            "branches": leaves.get(n, []), "cited": cited, "cited_detail": r["cited_in"],
            "link": (m or {}).get("link") or lg.get("link"), "pdf": (m or {}).get("pdf") or lg.get("pdf"),
            "scholar": "https://scholar.google.com/scholar?q=" + quote_plus(r["title"] or r["text"][:120]),
            "notes": f"public/notes/{key}.pdf" if os.path.exists(os.path.join(ROOT, "public", "notes", f"{key}.pdf")) else None,
            "notes_title": nmap.get(key, {}).get("file"),
            "visual": os.path.exists(os.path.join(ROOT, "papers", key, "visual.js")),
            "tags": meta.get("tags", []), "result": meta.get("result_plain") or "", "tabs": meta.get("tabs", []), "tag_tabs": meta.get("tag_tabs", {}),
            "thumb": next((f"public/thumbs/{key}.{e}" for e in ("webp", "png") if os.path.exists(os.path.join(ROOT, "public", "thumbs", f"{key}.{e}"))), None),
            "summary": lg.get("summary", ""), "assumptions": lg.get("assumptions", ""),
        })
    out = {"synced": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "source": S["source"], "survey": {"title": S["title"], "authors": S["authors"]},
           "taxonomy": [{"id": b["id"], "name": b["name"], "leaves": [lf["name"] for lf in b["leaves"]]} for b in S["taxonomy"]], "papers": papers}
    json.dump(out, open(os.path.join(ROOT, "data", "papers.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    nn = sum(1 for p in papers if p["notes"]); nv = sum(1 for p in papers if p["visual"]); nb = sum(1 for p in papers if p["notes"] and p["visual"])
    print(f"papers.json: {len(papers)} survey references, {nn} with notes, {nv} with visuals, {nb} complete")
    todo = [f"[{p['n']}] {p['key']}" for p in papers if p["notes"] and not p["visual"]]
    if todo:
        print("NEW WORK — notes but no visual yet:", ", ".join(todo))
    missing = [k for k in nmap if not os.path.exists(os.path.join(ROOT, "public", "notes", f"{k}.pdf"))]
    if missing:
        print("in notes_map.json but no PDF copied yet (run with --notes-dir):", missing)


if __name__ == "__main__":
    main()
