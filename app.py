"""
Spotify Customer Support AI Agent - Web Server
Lightweight server using Python standard library http.server (no external frameworks).
Serves the HTML/CSS/JS frontend and handles agent prediction requests.

Usage:
    python app.py
"""

import os
import sys
import json
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agent import SupportAgent

# Initialize agent once on server startup
print("Initializing SupportAgent and loading retrieval index...")
agent = SupportAgent()
print("SupportAgent ready!")

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


class SupportAgentHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean console logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def do_GET(self):
        # Map URL to local file in web/
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            file_path = os.path.join(WEB_DIR, "index.html")
        else:
            file_path = os.path.join(WEB_DIR, path.lstrip("/"))

        if os.path.exists(file_path) and os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "text/plain"

            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"Error reading file: {e}")
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        if self.path == "/api/process":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_len)
                body = json.loads(post_data.decode("utf-8"))

                customer_message = body.get("message", "").strip()
                if not customer_message:
                    self.send_error(400, "Missing 'message' field in JSON request")
                    return

                # Process query through agent pipeline
                result = agent.process_message(customer_message)

                # Format retrieved cases for frontend display
                retrieved_display = []
                for case in result.get("retrieved_cases", [])[:1]:
                    retrieved_display.append({
                        "case_id": case.get("case_id", ""),
                        "similarity": round(float(case.get("similarity", 0.0)), 4),
                        "customer_text": case.get("historical_issue", ""),
                        "brand_reply": case.get("historical_response", ""),
                    })

                response_payload = {
                    "intent": result.get("intent", ""),
                    "confidence": round(float(result.get("intent_confidence", 0.0)), 4),
                    "action": result.get("action", ""),
                    "reason": result.get("reason", ""),
                    "draft_reply": result.get("reply", ""),
                    "retrieved_cases": retrieved_display,
                }

                response_bytes = json.dumps(response_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)

            except Exception as e:
                error_bytes = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_bytes)))
                self.end_headers()
                self.wfile.write(error_bytes)
        else:
            self.send_error(404, "Endpoint not found")


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, SupportAgentHandler)
    print("=" * 65)
    print(f"  Spotify Support AI Agent Web UI running at:")
    print(f"  http://localhost:{port}")
    print("=" * 65)
    print("Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port)
