"""Static check: every relative path the site references (ES imports, fetch/src/href literals, :::survey images,
data files) exists in git with exactly that case. GitHub Pages is case-sensitive; Windows is not, so a wrong case
works on localhost and 404s live. Also lists tracked files that .gitignore would hide from the live site."""
import os, re, subprocess, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n"))
untracked = set(subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n"))
files = tracked | untracked
lower = {f.lower(): f for f in files}
lit = re.compile(r"""(?:import\s*(?:[\w{}\s,*]+from\s*)?|import\(|fetch\(|src=|href=|\(|\s)['"`]((?:\.{1,2}/|/)?[\w./-]+\.(?:js|json|md|css|webp|png|jpg|svg|pdf|html))(?:[?#][^'"`]*)?['"`]""")
bad = []
for f in sorted(files):
    if not f or not re.search(r"\.(js|html|md|json|css)$", f) or f.startswith(("scripts/", "research/", "reference/", "node_modules/", "data/")):
        continue
    s = open(os.path.join(ROOT, f), encoding="utf-8", errors="ignore").read()
    base = os.path.dirname(f)
    for m in lit.finditer(s):
        p = m.group(1)
        if "${" in p or p.startswith("//") or "/" not in p:
            continue
        tgt = os.path.normpath(p.lstrip("/") if p.startswith("/") else os.path.join(base, p)).replace("\\", "/")
        # page-relative links in html/js that are resolved against the site root (pages live at the root)
        cands = [tgt, os.path.normpath(p.lstrip("./")).replace("\\", "/")]
        if any(c in files for c in cands):
            continue
        hit = next((lower[c.lower()] for c in cands if c.lower() in lower), None)
        if hit:
            bad.append(f"{f}: '{p}' -> case mismatch, file is '{hit}'")
        elif any(c in untracked for c in cands):
            bad.append(f"{f}: '{p}' -> not committed")
        else:
            bad.append(f"{f}: '{p}' -> missing")
for b in bad:
    print(b)
print(f"{len(bad)} problems")
