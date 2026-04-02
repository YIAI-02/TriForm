import http.server, socketserver, os, webbrowser

PORT = int(os.environ.get("PORT", "8001"))
ROOT = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        new_path = http.server.SimpleHTTPRequestHandler.translate_path(self, path)
        rel = os.path.relpath(new_path, os.getcwd())
        return os.path.join(ROOT, rel)

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print(f"Serving {ROOT} at {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()
