import os
import sys
import json
import time
import socket
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Default directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(SCRIPT_DIR, "web")
CAPTURES_DIR = os.path.join(SCRIPT_DIR, "esp32_captures")
ANNOTATED_DIR = os.path.join(CAPTURES_DIR, "annotated")
RAW_DIR = os.path.join(CAPTURES_DIR, "raw")
STATUS_FILE = os.path.join(CAPTURES_DIR, "status.json")
LOG_FILE = os.path.join(CAPTURES_DIR, "events.log")

# Shared state in-memory cache
_shared_state = {
    "session": "IDLE",
    "score": 0,
    "grade": "No data",
    "durations": {"WORK": 0, "STUDY": 0, "BREAK": 0, "IDLE": 0},
    "objects": {},
    "latest_image": "",
    "last_capture_time": "",
    "recent_events": [],
    "past_sessions": []
}
_state_lock = threading.Lock()


def update_server_state(data: dict):
    """Called by dashboard.py to update the live shared state directly."""
    with _state_lock:
        _shared_state.update(data)
    # Also write to status.json as persistent cache
    try:
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(_shared_state, f)
    except Exception:
        pass


def get_latest_capture_path():
    """Find the most recent image in annotated/ or raw/ directory."""
    for folder in [ANNOTATED_DIR, RAW_DIR]:
        if os.path.exists(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                return max(files, key=os.path.getmtime)
    return None


def get_local_ip():
    """Determine the LAN Wi-Fi IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def parse_sessions_from_log():
    """Extract historical completed sessions from events.log."""
    if not os.path.exists(LOG_FILE):
        return []
    sessions = []
    current = None
    try:
        import datetime
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "Session Started:" in line:
                    try:
                        time_part = line[1:20]
                        sess_name = line.split("Session Started:")[1].strip()
                        current = {
                            "session": sess_name,
                            "start_time": time_part.split(" ")[1] if " " in time_part else time_part,
                            "full_start": time_part,
                            "date": time_part.split(" ")[0] if " " in time_part else ""
                        }
                    except Exception:
                        pass
                elif "Session Ended:" in line and current:
                    try:
                        time_part = line[1:20]
                        sess_name = line.split("Session Ended:")[1].strip()
                        end_time = time_part.split(" ")[1] if " " in time_part else time_part
                        dur = 0
                        try:
                            t1 = datetime.datetime.strptime(current["full_start"], "%Y-%m-%d %H:%M:%S")
                            t2 = datetime.datetime.strptime(time_part, "%Y-%m-%d %H:%M:%S")
                            dur = int((t2 - t1).total_seconds())
                        except Exception:
                            dur = 0
                        sessions.append({
                            "session": current["session"],
                            "start_time": current["start_time"],
                            "end_time": end_time,
                            "duration_seconds": max(dur, 0),
                            "date": current["date"]
                        })
                        current = None
                    except Exception:
                        pass
    except Exception as e:
        print(f"[Parse Sessions Error] {e}")
    return sessions


class DeskMonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence verbose request logs to keep terminal clean
        pass

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.serve_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
        elif path == "/style.css":
            self.serve_file(os.path.join(WEB_DIR, "style.css"), "text/css; charset=utf-8")
        elif path == "/app.js":
            self.serve_file(os.path.join(WEB_DIR, "app.js"), "application/javascript; charset=utf-8")
        elif path == "/api/status":
            self.serve_status()
        elif path == "/api/image":
            self.serve_latest_image()
        else:
            self.send_error(404, "Not Found")

    def serve_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, "File not found")
            return
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def serve_status(self):
        with _state_lock:
            state = dict(_shared_state)

        # If empty, attempt to read from status.json
        if not state.get("latest_image") and os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    state.update(cached)
            except Exception:
                pass

        # Populate latest image filename if missing
        if not state.get("latest_image"):
            latest_path = get_latest_capture_path()
            if latest_path:
                state["latest_image"] = os.path.basename(latest_path)
                state["last_capture_time"] = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(latest_path)))

        # Fallback for recent events from events.log if state is empty
        if not state.get("recent_events") and os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    state["recent_events"] = lines[-20:]
            except Exception:
                pass

        # If live past_sessions is empty, populate from parsed log
        if not state.get("past_sessions"):
            state["past_sessions"] = parse_sessions_from_log()[-30:]

        state["ip"] = get_local_ip()
        state["port"] = getattr(self.server, "server_port", 8080)

        payload = json.dumps(state).encode("utf-8")
        self.send_response(200)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(payload)

    def serve_latest_image(self):
        latest_path = get_latest_capture_path()
        if not latest_path or not os.path.exists(latest_path):
            # Create a 1x1 black fallback JPEG if no captures exist yet
            self.send_error(404, "No captures available yet")
            return

        try:
            with open(latest_path, "rb") as f:
                img_data = f.read()

            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img_data)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(img_data)
        except Exception as e:
            self.send_error(500, f"Error streaming image: {e}")


def run_server(port=8080, host="0.0.0.0"):
    local_ip = get_local_ip()
    print("=" * 60)
    print("  DESK ASSISTANT - MOBILE WEB MONITOR")
    print("=" * 60)
    print(f"  Local link (PC):       http://localhost:{port}")
    print(f"  Phone link (same Wi-Fi): http://{local_ip}:{port}")
    print("=" * 60)

    server = ThreadingHTTPServer((host, port), DeskMonitorHandler)
    server.server_port = port
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server.")
        server.server_close()


def start_web_server_thread(port=8080):
    """Spins up the web server in a background daemon thread."""
    local_ip = get_local_ip()
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), DeskMonitorHandler)
        server.server_port = port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[Mobile Monitor] Running on http://{local_ip}:{port}")
        return server, local_ip, port
    except Exception as e:
        print(f"[Mobile Monitor] Failed to start on port {port}: {e}")
        return None, local_ip, port


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)
