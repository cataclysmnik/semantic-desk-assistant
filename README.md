# Semantic Desk Assistant

An intelligent, privacy-first, 100% offline smart workspace monitor and productivity assistant. It pairs an **ESP32 wireless edge camera** with a local **Vision-Language Model (moondream2)** to understand what is on your desk, track your activity sessions (Work, Study, Break, Idle), and provide interactive visual question answering (VQA) without sending any images or telemetry to the cloud.

---

## Features

- **100% Offline & Private**: Runs entirely locally using `moondream2` (`transformers` + `torch`). Zero cloud calls, zero external API keys, zero visual data leaves your network.
- **Wireless IoT Edge Capture**: Automated snapshot polling over Wi-Fi HTTP from an ESP32-CAM or ESP32-S3 camera module.
- **Semantic Scene Understanding**: Intelligently identifies visible desk items (laptops, phones, books, notebooks, mugs, stationery) using small multimodal language reasoning instead of static bounding box heuristics.
- **Automated Productivity & Session Tracking**:
  - **WORK**: Open laptop / computer / keyboard in active use.
  - **STUDY**: Books, notebooks, notepad, or documents on the desk.
  - **BREAK / IDLE**: Absence of work items or empty desk area.
  - Live session timers, duration counters, and chronological event logs.
- **Contextual Visual Q&A (VQA)**: Ask natural language questions about your workspace (e.g., *"Where did I put my phone?"*, *"Describe what I'm working on"*). The assistant leverages both the current camera frame and recent detection history.
- **Dual Dashboard Interfaces**:
  - **Desktop GUI**: Native PyQt5 application with real-time video feed, session timeline graph, detection feed, and chat panel.
  - **Mobile Web Monitor**: Built-in HTTP server broadcasting a responsive, cyberpunk-styled dashboard (`web/index.html`) accessible on local mobile devices or secondary displays.

---

## System Architecture

```text
[ ESP32-CAM / ESP32-S3 ]
       │  (Wi-Fi HTTP snapshot endpoint)
       ▼
[ PyQt5 Dashboard (dashboard.py) ]
  ├── QNetworkAccessManager (Periodic / manual frame capture)
  ├── VlDetectionWorker (moondream2 scene query & synonym extractor)
  ├── Session State Engine (WORK / STUDY / BREAK / IDLE)
  ├── VqaWorker (Context-aware Visual Q&A thread)
  └── Mobile Server Thread (Broadcasts telemetry to web/app.js)
       │
       ▼
[ Mobile / Desktop Browser UI (web/) ]
```

---

## Directory Structure

```text
semantic-desk-assistant/
├── dashboard.py                  # Main PyQt5 application, model runner, & GUI
├── moondream2/                   # Local offline moondream2 model weights & tokenizer
├── web/                          # Mobile/browser dashboard web app
│   ├── index.html                # Cyberpunk-themed telemetry interface
│   ├── app.js                    # Web UI logic, session timer, & event feeds
│   └── style.css                 # Custom cyberpunk styling tokens
├── web_server.py                 # Standalone web/telemetry server
├── esp32_camera/                 # ESP32-CAM Arduino sketch firmware
├── freenova_esp32s3_cam...ino    # ESP32-S3 camera configuration sketch
├── app_httpd.cpp                 # Camera web server backend (ESP32)
├── camera_pins.h                 # Camera pin definitions (ESP32-CAM / ESP32-S3)
├── esp32_captures/               # Local cache of raw and processed frames
└── requirements.txt              # Project dependencies
```

---

## Getting Started

### 1. Prerequisites

- **Python**: 3.10 – 3.12 (or 3.13)
- **PyTorch**: Recommended with CUDA if GPU is available, or CPU mode.
- **Hardware**:
  - ESP32-CAM or Freenova ESP32-S3 CAM module.
  - Local workstation (Windows, Linux, or macOS).

### 2. Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/your-username/semantic-desk-assistant.git
cd semantic-desk-assistant

python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

Install core dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or standard pip install torch
pip install "transformers==4.46.3" PyQt5 pillow pyqtgraph requests
```

> **Note on `transformers`**: `moondream2` relies on `transformers` (v4.44 – v4.46 recommended) for full compatibility with its custom modeling architecture.

### 3. Flash the ESP32 Camera

1. Open `esp32_camera/esp32_camera.ino` or `freenova_esp32s3_cam_setup_and_webserver.ino` in the Arduino IDE.
2. Select your board (e.g. `AI Thinker ESP32-CAM` or `ESP32S3 Dev Module`).
3. Set your Wi-Fi SSID and password in the sketch.
4. Upload firmware to the board and open the Serial Monitor (115200 baud) to obtain the assigned IP address (e.g., `http://192.168.1.100`).

### 4. Run the Dashboard

Launch the assistant from the project root:

```bash
python dashboard.py
```

1. Enter your ESP32 Camera IP/URL in the top bar.
2. Click **Connect / Start Stream**.
3. The dashboard will automatically fetch captures, run offline `moondream2` inference, update the session timeline, and display detected objects.
4. View the mobile monitor on your phone by opening the displayed local URL (e.g. `http://<YOUR_LOCAL_IP>:8080`).

---

## Vision-Language Capabilities

### Offline Object Recognition
The assistant queries the scene with focused prompts:
```python
prompt = "What objects are on the desk?"
answer = model.answer_question(enc_image, prompt, tokenizer=tokenizer)
```
The model describes detected objects in natural language (e.g., *"On the desk, there is a laptop, a cell phone, and a book."*), which the assistant maps into canonical categories to determine active sessions.

### Contextual VQA
Users can chat with the assistant in real-time. The prompt is augmented with chronological detection history:
```text
Context — here is a log of recent object detection events (Past -> Present):
[23:05:10] Detected: 1x laptop
[23:09:18] Detected: 1x cell phone
[23:14:02] Detected: 1x mug

Question: Did I leave my coffee cup on the desk?
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
