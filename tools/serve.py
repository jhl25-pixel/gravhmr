"""Static server with HTTP Range support (video seeking needs it; python -m http.server has none).
  python tools/serve.py [port] [dir]"""
import os, sys, re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class H(SimpleHTTPRequestHandler):
    def send_head(self):
        r = self.headers.get("Range"); path = self.translate_path(self.path)
        if not r or not os.path.isfile(path): return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", r); size = os.path.getsize(path)
        a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2))); b = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        b = min(b, size - 1)
        if a > b: self.send_error(416); return None
        f = open(path, "rb"); f.seek(a); self._left = b - a + 1
        self.send_response(206); self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {a}-{b}/{size}"); self.send_header("Content-Length", str(self._left))
        self.send_header("Accept-Ranges", "bytes"); self.end_headers(); return f
    def copyfile(self, src, dst):
        n = getattr(self, "_left", None)
        if n is None: return super().copyfile(src, dst)
        while n > 0:
            c = src.read(min(65536, n))
            if not c: break
            dst.write(c); n -= len(c)
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes"); self.send_header("Cache-Control", "no-cache"); super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
os.chdir(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs"))
ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
