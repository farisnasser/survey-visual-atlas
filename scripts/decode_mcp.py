"""Decode Google Drive connector downloads into real files.

The Drive connector's download_file_content returns base64 JSON; when the result is large, Claude Code saves it as
<session>/tool-results/mcp-...-download_file_content-<ts>.txt. This script decodes every such file in a folder and
writes it under its Drive title. Parallel downloads can collide on one filename, so always check the printed titles
against what you requested and re-download anything missing.

usage: python scripts/decode_mcp.py <tool-results-dir> <out-dir> [--keep]
"""
import base64
import glob
import json
import os
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    keep = "--keep" in sys.argv
    os.makedirs(out, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(src, "*download_file_content*.txt"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # not a JSON payload
            print(f"skip {os.path.basename(f)}: {e}")
            continue
        title = d.get("title")
        if not title or "content" not in d:
            continue
        p = os.path.join(out, title)
        with open(p, "wb") as fh:
            fh.write(base64.b64decode(d["content"]))
        print(f"{title}  ({os.path.getsize(p) // 1024} KB, {d.get('mimeType')})")
        n += 1
        if not keep:
            os.remove(f)
    print(f"decoded {n} file(s) into {out}")


if __name__ == "__main__":
    main()
