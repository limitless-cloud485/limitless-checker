"""HTTP wrapper for the Limitless footage checker (Render/Koyeb/Docker).

POST /verify  {"line": "...", "candidates": [{"url","title"}, ...]}
  -> {"line": ..., "results": [{"url","title","score","neg","band"}...]}
GET  /health  -> {"ok": true}
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app import verify


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/verify"):
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            out = verify(payload.get("line", ""), json.dumps(payload.get("candidates", [])))
            self._json(200, json.loads(out))
        except Exception as e:
            self._json(500, {"error": str(e)[:200]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8791"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
