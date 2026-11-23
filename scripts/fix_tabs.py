"""Repair LaTeX commands mangled into control characters by shell heredocs (\t -> TAB, \b -> BS, \f -> FF, \r, \v, \a)."""
import sys, re
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read(); s0 = s
    for ch, rep in (("\t", "\\t"), ("\b", "\\b"), ("\f", "\\f"), ("\v", "\\v"), ("\a", "\\a")):
        s = re.sub(re.escape(ch) + r"(?=[a-zA-Z])", lambda m: rep, s)
    if s != s0:
        open(p, "w", encoding="utf-8").write(s); print("repaired", p)
