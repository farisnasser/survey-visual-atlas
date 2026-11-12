"""Old -> new numbering of survey v3 after the paragraph-by-paragraph rewrite (2026-11-12).

old: survey-v3/tex/main.aux (the build the guide was written against, before the rewrite) and the 80-entry bibliography
     saved before references were dropped/reordered (passed as argv[1]).
new: a fresh build of survey-v3/overleaf/main.tex (aux kept in a temp dir) and survey-v3/tex/refs.tex.
Writes research/survey3_renumber.json: {"eq": {old: new|null}, "ref": {old: new|null}, "fig": ..., "tab": ..., "labels": {...}}."""
import re, os, sys, json, subprocess, shutil, tempfile
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SV = os.path.join(ROOT, "..", "survey-v3")
TEC = r"C:\Users\Asus\tools\tectonic\tectonic.exe"

def labels(aux):
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{(.*?)\}\{", aux):
        v = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", m.group(2)).strip("{} ")
        out[m.group(1)] = v
    return out

old = labels(open(os.path.join(SV, "tex", "main.aux"), encoding="utf-8").read())
tmp = tempfile.mkdtemp()
shutil.copytree(os.path.join(SV, "overleaf"), os.path.join(tmp, "o"))
subprocess.run([TEC, "-X", "compile", "--keep-intermediates", "main.tex"], cwd=os.path.join(tmp, "o"), capture_output=True)
new = labels(open(os.path.join(tmp, "o", "main.aux"), encoding="utf-8").read())

def keys(path):
    return re.findall(r"\\bibitem\{([^}]+)\}", open(path, encoding="utf-8").read())
ko, kn = keys(sys.argv[1]), keys(os.path.join(SV, "tex", "refs.tex"))
res = {"eq": {}, "fig": {}, "tab": {}, "sec": {}, "ref": {}, "labels": {}}
for lab, v in old.items():
    kind = lab.split(":")[0]
    if kind in ("eq", "fig", "tab"):
        res[kind][v] = new.get(lab)
        res["labels"][lab] = [v, new.get(lab)]
for i, k in enumerate(ko):
    res["ref"][str(i + 1)] = (kn.index(k) + 1) if k in kn else None
res["refkeys_new"] = {str(i + 1): k for i, k in enumerate(kn)}
res["new_labels"] = {l: v for l, v in new.items() if l not in old}
json.dump(res, open(os.path.join(ROOT, "research", "survey3_renumber.json"), "w"), indent=1)
print("eqs old", len(res["eq"]), "dropped", sum(v is None for v in res["eq"].values()))
print("dropped eqs:", sorted([int(k) for k, v in res["eq"].items() if v is None]))
print("fig changes:", {k: v for k, v in res["fig"].items() if k != v}, "tab changes:", {k: v for k, v in res["tab"].items() if k != v})
print("dropped refs:", [k for k, v in res["ref"].items() if v is None])
