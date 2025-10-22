import http.server
import socketserver
import threading
from pathlib import Path
import toga
from toga import App, MainWindow, WebView

PORT = 8000
# Ensure the file exists
html_path = Path("index.html")
if not html_path.is_file():
    raise FileNotFoundError(f"{html_path} not found in the current folder")


class SingleFileHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Always serve the same HTML file regardless of the request path
        self.path = "/"
        self.path += str(html_path.name)
        return super().do_GET()


class WebApp(App):
    def startup(self):
        self.main_window = MainWindow(title="Inference Demo")
        self.webview = WebView(url=f"http://localhost:{PORT}")
        self.main_window.content = self.webview
        self.main_window.show()
        with socketserver.TCPServer(("", PORT), SingleFileHandler) as httpd:
            print(f"Serving {html_path.name} at http://localhost:{PORT}")
            httpd.serve_forever()


def start_http():
    with socketserver.TCPServer(("", PORT), SingleFileHandler) as httpd:
        print(f"Serving {html_path.name} at http://localhost:{PORT}")
        httpd.serve_forever()


http_thread = threading.Thread(target=start_http, daemon=True)
http_thread.start()


WebApp("webapp", "org.example.ViewWindow").main_loop()
