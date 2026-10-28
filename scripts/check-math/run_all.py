"""Run every per-paper math check and write results.txt next to this file.

usage: python scripts/check-math/run_all.py   (takes ~10 minutes: Safran, Wu and Allen-Zhu replicate experiments)
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
scripts = sorted(f for f in glob.glob(os.path.join(HERE, "*.py")) if os.path.basename(f) not in ("common.py", "run_all.py"))
out, failed = [], []
for s in scripts:
    p = subprocess.run([sys.executable, s], cwd=HERE, capture_output=True, text=True)
    out.append(p.stdout + p.stderr)
    if p.returncode:
        failed.append(os.path.basename(s))
    print(f"{'FAIL' if p.returncode else 'ok  '} {os.path.basename(s)}")
with open(os.path.join(HERE, "results.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("all passed" if not failed else f"failed: {failed}")
sys.exit(1 if failed else 0)
