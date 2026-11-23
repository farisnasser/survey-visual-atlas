"""Print the final (submitted, no-appendix) survey's equation, figure, table and section numbers.
usage: python survey_final_labels.py <dir with main.aux>"""
import re, sys, os
sys.stdout.reconfigure(encoding="utf-8")
a = open(os.path.join(sys.argv[1], "main.aux"), encoding="utf-8").read()
L = []
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{(.*?)\}\{", a):
    v = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", m.group(2)).strip("{} ")
    L.append((m.group(1), v))
for kind in ("eq", "fig", "tab", "sec", "alg"):
    print(kind + ":", ", ".join(f"{k.split(':', 1)[1]}={v}" for k, v in L if k.startswith(kind + ":")))
