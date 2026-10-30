"""Build data/survey.json from the final survey PDF: the reference list, where each reference is cited, and the Fig. 5 taxonomy.

The survey (NonConvex_Survey_Final.pdf, Nasser & Moussa) is the source of truth for which papers the Atlas covers.
Run again only when the survey PDF changes.

usage: python scripts/build_survey.py --pdf <NonConvex_Survey_Final.pdf>
"""
import argparse
import json
import os
import re

import fitz  # PyMuPDF

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Fig. 5, transcribed from the PDF (checked against the extracted figure text below)
TAXONOMY = [
    ["III", "Geometry of critical points", [["Linear & deep linear", "4,31,52"], ["Saddle proliferation", "5,38"], ["Non-linear nets", "53-57"],
                                            ["Low-rank testbeds", "6,58-60"], ["Higher-order / empirical", "61-63"]]],
    ["IV", "Dynamics of GD/SGD near saddles", [["Asymptotic avoidance", "7,43,64-67"], ["Escape time", "68"], ["Plateaus, saddle-to-saddle", "18,30,56"],
                                                ["Noise as mechanism", "6,69-72"]]],
    ["V", "Algorithms designed to escape saddles", [["Second-order / cubic", "5,28,40,73-77"], ["Perturbation", "8,78,79"], ["Acceleration", "9,10,80"],
                                                    ["Hessian-free negative curvature", "33,81,82"], ["Variance reduction & bounds", "11,12,47,83-88"], ["Adaptive", "66,89"]]],
    ["VI", "Overparameterization", [["NTK / lazy", "13-15,90-92"], ["Mean field", "32,93,94"], ["Beyond lazy & costs", "57,95,96"]]],
    ["VII", "Dynamics of practical training", [["Init. & momentum", "29,97"], ["Large LR & edge of stability", "16,17,35-37,39,98-100"],
                                               ["Sharpness-aware", "34,101,102"], ["Modern optimizers", "42,103-107"]]],
]


def nums(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


KEEP = {"non", "first", "second", "third", "higher", "high", "low", "large", "small", "over", "under", "self", "multi", "mean", "saddle",
        "edge", "cross", "fine", "well", "pre", "post", "semi", "sharpness", "gradient", "two", "three", "zeroth", "heavy", "loss", "quasi", "online"}
KEEP_AFTER = ("free", "order", "based", "like", "aware", "point", "layer", "scale", "step", "tail", "field", "and")


def dehyphen(s):
    """Undo line-break hyphenation ("Repre-<newline>sentations" -> "Representations"), keeping real compounds and names."""
    keep = lambda a, b: a.lower() in KEEP or b.lower().startswith(KEEP_AFTER) or b[:1].isupper()
    return re.sub(r"(\w+)-\n(\w+)", lambda m: m.group(1) + ("-" if keep(m.group(1), m.group(2)) else "") + m.group(2), s)


def roman_heading(line):
    m = re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\. ([A-Z][A-Z :,’'\-–/]+)$", line.strip())
    return m.group(0).strip() if m else None


def parse(pdf):
    doc = fitz.open(pdf)
    text = "\n".join(p.get_text() for p in doc)
    body, _, refs_txt = text.partition("\nREFERENCES\n")
    refs = {}
    for m in re.finditer(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]\s|\Z)", refs_txt, re.S):
        refs[int(m.group(1))] = re.sub(r"\s+", " ", dehyphen(m.group(2))).strip()
    # drop the running header if it leaked into the last entries
    refs = {k: re.sub(r"MLR 570 SURVEY.*$", "", v).strip() for k, v in refs.items()}
    cites, sec, sub = {}, "Front matter", ""
    for line in body.split("\n"):
        h = roman_heading(line)
        if h:
            sec, sub = h, ""
            continue
        m = re.match(r"^([A-H])\. [A-Z]", line.strip())
        if m and len(line) < 90 and sec != "Front matter" and " are with " not in line:
            sub = line.strip().rstrip("-")
        for grp in re.finditer(r"\[(\d+)\](?:\s*[–-]\s*\[(\d+)\])?", line):
            a, b = int(grp.group(1)), int(grp.group(2) or grp.group(1))
            for n in range(a, b + 1):
                cites.setdefault(n, [])
                if [sec, sub] not in cites[n]:
                    cites[n].append([sec, sub])
    fig = text[text.find("Non-convex\noptimization\nin neural"):text.find("Fig. 5.")]
    return refs, cites, fig


def split_ref(s):
    """IEEE reference -> authors, title, venue, year. Books have no quoted title: 'A. Author, Title. Publisher, 2004.'"""
    m = re.match(r"^(.*?),?\s*“(.+?),?”\s*(.*)$", s)
    if m:
        au, ti, rest = m.groups()
    else:
        mb = re.match(r"^(.*?[A-Z][a-z]+), (.+?)\. (.*)$", s)
        if not mb:
            return {"authors": s, "title": "", "venue": "", "year": None}
        au, ti, rest = mb.groups()
    clean = re.sub(r"arXiv:\S+", "", rest)
    yrs = re.findall(r"\b(?:19|20)\d\d\b", clean)
    year = int(yrs[-1]) if yrs else None
    venue = re.sub(r"^in\s+", "", rest).strip(" .,")
    venue = re.sub(r",?\s*(?:19|20)\d\d\.?$", "", venue).strip(" .,")
    venue = re.sub(r"^(?:19|20)\d\d,\s*", "", venue)   # "2020, arXiv:2003.02218" -> "arXiv:2003.02218"
    return {"authors": au.strip(), "title": ti.strip().rstrip(","), "venue": venue, "year": year}


def resolve_dashes(refs):
    """IEEE style writes "——" for the same authors as the previous reference."""
    prev = None
    for k in sorted(refs, key=int):
        if refs[k]["authors"].strip("—–- ") == "" and prev:
            refs[k]["authors"] = prev
        prev = refs[k]["authors"]
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    a = ap.parse_args()
    refs, cites, fig = parse(a.pdf)
    figrefs = sorted({int(x) for x in re.findall(r"\[(\d+)\]", fig)} | {n for lo, hi in re.findall(r"\[(\d+)\]–\[(\d+)\]", fig) for n in range(int(lo), int(hi) + 1)})
    tax_refs = sorted({n for _, _, leaves in TAXONOMY for _, spec in leaves for n in nums(spec)})
    assert figrefs == tax_refs, f"Fig. 5 transcription differs from the PDF: {set(figrefs) ^ set(tax_refs)}"
    out = {
        "source": os.path.basename(a.pdf),
        "title": "Non-Convex Optimization in Neural Networks: Saddles, Escape Methods and Practical Dynamics",
        "authors": "Faris Nasser & Salah Moussa",
        "refs": resolve_dashes({str(k): {"text": v, **split_ref(v), "cited_in": cites.get(k, [])} for k, v in sorted(refs.items())}),
        "taxonomy": [{"id": i, "name": n, "leaves": [{"name": ln, "refs": nums(spec)} for ln, spec in leaves]} for i, n, leaves in TAXONOMY],
    }
    json.dump(out, open(os.path.join(ROOT, "data", "survey.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    uncited = [k for k in refs if k not in cites]
    print(f"survey.json: {len(refs)} references, {len(tax_refs)} in the Fig. 5 taxonomy, never cited in the text: {uncited}")


if __name__ == "__main__":
    main()
