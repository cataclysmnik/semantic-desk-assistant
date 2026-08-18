import sys
import os
import time
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QGraphicsDropShadowEffect, QFrame, QComboBox,
                               QPlainTextEdit, QSplitter)
from PySide6.QtCore import Qt, QTimer, QUrl, QThread, Signal
from PySide6.QtGui import QPixmap, QFont, QColor, QTextCursor
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics (pip install ultralytics) to use YOLO features.")
    YOLO = None

class YoloWorker(QThread):
    # Emits annotated image path and a dictionary of detected objects {class_name: count}
    finished_processing = Signal(str, dict)
    error_processing = Signal(str)

    def __init__(self, raw_path, annotated_dir, model):
        super().__init__()
        self.raw_path = raw_path
        self.annotated_dir = annotated_dir
        self.model = model

    def run(self):
        try:
            if not self.model:
                raise Exception("YOLO model is not loaded.")
                
            # Run inference
            results = self.model(self.raw_path, verbose=False)
            
            # Count detected objects
            detected_counts = {}
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
            
            # results[0].plot() returns BGR image numpy array
            annotated_img = results[0].plot()
            
            # Save annotated image
            filename = os.path.basename(self.raw_path)
            annotated_path = os.path.join(self.annotated_dir, filename)
            cv2.imwrite(annotated_path, annotated_img)
            
            self.finished_processing.emit(annotated_path, detected_counts)
        except Exception as e:
            self.error_processing.emit(str(e))


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 Cam & YOLO Dashboard")
        self.setMinimumSize(1000, 700)
        
        # --- Directories ---
        self.raw_dir = os.path.join("esp32_captures", "raw")
        self.annotated_dir = os.path.join("esp32_captures", "annotated")
        self.log_file = os.path.join("esp32_captures", "events.log")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)

        # --- YOLO Model & State ---
        self.yolo_model = YOLO("yolov8m.pt") if YOLO else None
        self.previous_objects = {}  # Tracks {class_name: count}
        
        # --- Camera State ---
        self.is_running = False
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.on_image_downloaded)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fetch_image)
        self.timer.setInterval(10000)

        self.setup_ui()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.central_widget.setStyleSheet("""
            QWidget {
                background-color: #F8F9FA;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
            }
            QLabel {
                color: #2D3436;
            }
            QLineEdit, QComboBox {
                padding: 8px 12px;
                border: 1px solid #DFE6E9;
                border-radius: 6px;
                background-color: #FFFFFF;
                font-size: 13px;
                color: #2D3436;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #74B9FF;
            }
            QPushButton {
                padding: 8px 16px;
                background-color: #0984E3;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #74B9FF;
            }
            QPushButton:pressed {
                background-color: #005691;
            }
            QPushButton#stopBtn {
                background-color: #D63031;
            }
            QPushButton#stopBtn:hover {
                background-color: #FF7675;
            }
            QPlainTextEdit {
                background-color: #FFFFFF;
                color: #2D3436;
                border: 1px solid #DFE6E9;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)

        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title_label = QLabel("ESP32 Control Center")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # --- Left Panel: Camera Stream ---
        camera_panel = QWidget()
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(0, 0, 10, 0)

        cam_title = QLabel("Auto Capture + YOLO (10s)")
        cam_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        camera_layout.addWidget(cam_title)

        cam_controls = QHBoxLayout()
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("ESP32 IP e.g. 192.168.137.28")
        self.ip_input.setText("192.168.137.28")
        
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(["0°", "90°", "180°", "270°"])
        self.rotation_combo.setToolTip("Rotate incoming image clockwise")
        
        self.toggle_btn = QPushButton("Start Auto Capture")
        self.toggle_btn.clicked.connect(self.toggle_capture)
        
        cam_controls.addWidget(self.ip_input)
        cam_controls.addWidget(self.rotation_combo)
        cam_controls.addWidget(self.toggle_btn)
        camera_layout.addLayout(cam_controls)

        self.capture_info_label = QLabel("Ready to capture")
        self.capture_info_label.setStyleSheet("color: #636E72; font-size: 12px;")
        camera_layout.addWidget(self.capture_info_label)

        if not self.yolo_model:
            yolo_warning = QLabel("WARNING: YOLO is disabled. 'ultralytics' is not installed.")
            yolo_warning.setStyleSheet("color: #D63031; font-size: 12px; font-weight: bold;")
            camera_layout.addWidget(yolo_warning)

        self.image_frame = QFrame()
        self.image_frame.setStyleSheet("QFrame { background-color: #FFFFFF; border-radius: 10px; border: 1px solid #DFE6E9; }")
        shadow1 = QGraphicsDropShadowEffect()
        shadow1.setBlurRadius(15)
        shadow1.setColor(QColor(0, 0, 0, 15))
        shadow1.setOffset(0, 3)
        self.image_frame.setGraphicsEffect(shadow1)

        image_layout = QVBoxLayout(self.image_frame)
        self.image_label = QLabel("No image yet")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #B2BEC3; font-size: 16px;")
        image_layout.addWidget(self.image_label)
        
        camera_layout.addWidget(self.image_frame, stretch=1)
        splitter.addWidget(camera_panel)

        # --- Right Panel: Events Log ---
        events_panel = QWidget()
        events_layout = QVBoxLayout(events_panel)
        events_layout.setContentsMargins(10, 0, 0, 0)

        events_title = QLabel("Events Log")
        events_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        events_layout.addWidget(events_title)

        self.events_output = QPlainTextEdit()
        self.events_output.setReadOnly(True)
        
        shadow2 = QGraphicsDropShadowEffect()
        shadow2.setBlurRadius(15)
        shadow2.setColor(QColor(0, 0, 0, 15))
        shadow2.setOffset(0, 3)
        self.events_output.setGraphicsEffect(shadow2)

        events_layout.addWidget(self.events_output, stretch=1)
        
        clear_btn = QPushButton("Clear Events")
        clear_btn.clicked.connect(self.events_output.clear)
        events_layout.addWidget(clear_btn)

        splitter.addWidget(events_panel)
        
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)

    # --- Camera Stream Methods ---
    def toggle_capture(self):
        if not self.is_running:
            ip = self.ip_input.text().strip()
            if not ip:
                self.image_label.setText("Please enter a valid IP address.")
                return
                
            self.url = f"http://{ip}/capture"
            self.is_running = True
            self.toggle_btn.setText("Stop Auto Capture")
            self.toggle_btn.setObjectName("stopBtn")
            self.toggle_btn.style().unpolish(self.toggle_btn)
            self.toggle_btn.style().polish(self.toggle_btn)
            
            self.capture_info_label.setText("Fetching first image...")
            self.fetch_image()
            self.timer.start()
        else:
            self.is_running = False
            self.timer.stop()
            self.toggle_btn.setText("Start Auto Capture")
            self.toggle_btn.setObjectName("")
            self.toggle_btn.style().unpolish(self.toggle_btn)
            self.toggle_btn.style().polish(self.toggle_btn)
            self.capture_info_label.setText("Auto Capture paused")
            self.log_event("Capture paused.")

    def fetch_image(self):
        if not self.is_running:
            return
        request = QNetworkRequest(QUrl(self.url))
        self.network_manager.get(request)

    def on_image_downloaded(self, reply):
        if not self.is_running:
            reply.deleteLater()
            return
            
        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = reply.readAll()
            
            try:
                # Decode the raw bytes into an OpenCV image
                nparr = np.frombuffer(data.data(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Apply selected rotation
                    rotation = self.rotation_combo.currentText()
                    if rotation == "90°":
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == "180°":
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    elif rotation == "270°":
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    raw_filename = os.path.join(self.raw_dir, f"capture_{timestamp}.jpg")
                    
                    # Save the (possibly rotated) raw image
                    cv2.imwrite(raw_filename, img)
                    
                    if self.yolo_model:
                        self.capture_info_label.setText(f"Running YOLO on {timestamp}...")
                        
                        self.worker = YoloWorker(raw_filename, self.annotated_dir, self.yolo_model)
                        self.worker.finished_processing.connect(self.on_yolo_finished)
                        self.worker.error_processing.connect(self.on_yolo_error)
                        self.worker.start()
                    else:
                        self.capture_info_label.setText(f"Last capture: {time.strftime('%H:%M:%S')} - Saved to {self.raw_dir}")
                        self.show_image_from_path(raw_filename)
                else:
                    raise Exception("Failed to decode image data.")

            except Exception as e:
                self.capture_info_label.setText(f"Failed to process image: {e}")
        else:
            self.image_label.setText(f"Connection error: {reply.errorString()}")
            self.capture_info_label.setText("Retrying in 10s...")
            self.log_event(f"Error: {reply.errorString()}")
            
        reply.deleteLater()

    def on_yolo_finished(self, annotated_path, current_objects):
        self.capture_info_label.setText(f"Last capture: {time.strftime('%H:%M:%S')} - Saved raw & annotated")
        self.show_image_from_path(annotated_path)
        
        # Compare current_objects with previous_objects to detect events
        all_classes = set(current_objects.keys()).union(set(self.previous_objects.keys()))
        events = []
        
        for cls in all_classes:
            current_count = current_objects.get(cls, 0)
            prev_count = self.previous_objects.get(cls, 0)
            
            diff = current_count - prev_count
            if diff > 0:
                events.append(f"🟢 Detected: {diff}x {cls}")
            elif diff < 0:
                events.append(f"🔴 Removed: {abs(diff)}x {cls}")
                
        if events:
            for event in events:
                self.log_event(event)
                
        self.previous_objects = current_objects

    def on_yolo_error(self, err_msg):
        self.capture_info_label.setText(f"YOLO Error: {err_msg}")
        self.log_event(f"YOLO Error: {err_msg}")

    def show_image_from_path(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("Error loading image from disk")

    def log_event(self, message):
        short_time = time.strftime("[%H:%M:%S]")
        full_time = time.strftime("[%Y-%m-%d %H:%M:%S]")
        
        # Update UI
        self.events_output.appendPlainText(f"{short_time} {message}")
        self.events_output.moveCursor(QTextCursor.End)
        
        # Append to log file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{full_time} {message}\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Dashboard()
    window.show()
    sys.exit(app.exec())
