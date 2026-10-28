"""Local dev server for the Atlas: serves the repo root and accepts POST /__thumb?id=<Key> (a PNG data URL)
to save public/thumbs/<Key>.webp. Open http://localhost:8570/thumbs.html to regenerate every thumbnail.

usage: python scripts/serve.py [port]
"""
import base64
import http.server
import os
import re
import sys
import urllib.parse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8570


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/__thumb":
            self.send_error(404)
            return
        key = urllib.parse.parse_qs(u.query).get("id", [""])[0]
        if not re.fullmatch(r"[a-z0-9]+", key):
            self.send_error(400, "bad key")
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        data = base64.b64decode(body.split(",", 1)[1])
        os.makedirs(os.path.join(ROOT, "public", "thumbs"), exist_ok=True)
        ext = "webp" if body.startswith("data:image/webp") else "png"
        with open(os.path.join(ROOT, "public", "thumbs", f"{key}.{ext}"), "wb") as fh:
            fh.write(data)
        self.send_response(204)
        self.end_headers()
        print(f"saved thumb {key} ({len(data) // 1024} KB)")


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
