import sys
import os
import time
import cv2
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QGraphicsDropShadowEffect, QFrame, QComboBox,
                               QPlainTextEdit, QSplitter, QCheckBox, QFileDialog,
                               QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QTimer, QUrl, QThread, Signal, QSettings
from PySide6.QtGui import QPixmap, QFont, QColor, QTextCursor
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply



try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("Please install transformers and torch (pip install transformers torch pillow) for AI Chat.")
    Qwen2VLForConditionalGeneration = None
    AutoProcessor = None


class ModelLoader(QThread):
    models_loaded = Signal(object, object)
    error_loading = Signal(str)
    
    def run(self):
        try:
            if not Qwen2VLForConditionalGeneration:
                raise Exception("Transformers library not found.")
                
            model_id = "Qwen/Qwen2-VL-2B-Instruct"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch_dtype,
            ).to(device)
            processor = AutoProcessor.from_pretrained(model_id)
                
            self.models_loaded.emit(model, processor)
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            self.error_loading.emit(f"{str(e)}\n\nTraceback:\n{full_traceback}")


class VlDetectionWorker(QThread):
    finished_processing = Signal(str, dict)
    error_processing = Signal(str)

    def __init__(self, raw_path, annotated_dir, model, processor):
        super().__init__()
        self.raw_path = raw_path
        self.annotated_dir = annotated_dir
        self.model = model
        self.processor = processor

    def run(self):
        try:
            if not self.model:
                raise Exception("VL model is not loaded.")
            
            image = Image.open(self.raw_path).convert("RGB")
            device = next(self.model.parameters()).device

            prompt = 'List all main objects in this image and their counts. Return ONLY a valid JSON dictionary like {"object_name": count}. Do not output any other text or markdown. Example: {"laptop": 1, "cup": 2}'
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an object detection AI. You must output strictly in JSON format. Do not include markdown formatting or any other text."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt",
            ).to(device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            answer = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            import json
            import shutil
            
            json_str = answer.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
                
            try:
                detected_counts = json.loads(json_str.strip())
            except json.JSONDecodeError:
                detected_counts = {}
                
            # Copy raw image to annotated dir for display (we skip drawing boxes)
            filename = os.path.basename(self.raw_path)
            annotated_path = os.path.join(self.annotated_dir, filename)
            shutil.copy(self.raw_path, annotated_path)
            
            self.finished_processing.emit(annotated_path, detected_counts)
        except Exception as e:
            self.error_processing.emit(str(e))


class VqaWorker(QThread):
    finished_processing = Signal(str)
    error_processing = Signal(str)

    def __init__(self, raw_path, prompt, model, processor, context_log=""):
        super().__init__()
        self.raw_path = raw_path
        self.prompt = prompt
        self.model = model
        self.processor = processor
        self.context_log = context_log

    def run(self):
        try:
            image = Image.open(self.raw_path).convert("RGB")
            device = next(self.model.parameters()).device

            system_content = "You are a helpful AI vision assistant. Answer the user's questions about the image directly and concisely. If an object is not in the image, say so."
            if self.context_log.strip():
                system_content += f"\n\nHere is a log of recent object detection events tracking changes over time (Past -> Present):\n{self.context_log}\nUse this log to answer questions about what changed or happened. If an object was removed, you MUST mention the exact time it was removed according to the log."
                
            messages = [
                {
                    "role": "system", 
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt",
            ).to(device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            answer = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            self.finished_processing.emit(answer)
        except Exception as e:
            self.error_processing.emit(str(e))


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Semantic Desk Assistant")
        self.setMinimumSize(1200, 750)
        
        self.settings = QSettings("Sagni", "SemanticDeskAssistant")
        
        self.raw_dir = os.path.join("esp32_captures", "raw")
        self.annotated_dir = os.path.join("esp32_captures", "annotated")
        self.log_file = os.path.join("esp32_captures", "events.log")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)


        self.previous_objects = {}
        
        self.vl_model = None
        self.vl_processor = None
        self.latest_raw_image = None
        
        self.is_running = False
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.on_image_downloaded)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fetch_image)
        self.timer.setInterval(10000)

        self.setup_ui()
        
        # Start loading Qwen2-VL in the background
        if Qwen2VLForConditionalGeneration:
            self.chat_output.appendPlainText("[System] Loading Qwen2-VL-2B model in the background... This may take a moment.")
            self.model_loader = ModelLoader()
            self.model_loader.models_loaded.connect(self.on_models_loaded)
            self.model_loader.error_loading.connect(self.on_models_load_error)
            self.model_loader.start()
        else:
            self.chat_output.appendPlainText("[System] Please install transformers to use AI chat.")
            self.chat_input.setDisabled(True)
            self.send_btn.setDisabled(True)

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.central_widget.setStyleSheet("""
            QWidget {
                background-color: #F5F5F5;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QLabel { color: #000000; }
            QCheckBox { color: #000000; font-weight: bold; font-size: 13px; text-transform: uppercase; }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #000000;
                background-color: #FFFFFF;
            }
            QCheckBox::indicator:checked {
                background-color: #E50000;
            }
            QLineEdit, QComboBox {
                padding: 8px 12px;
                border: 2px solid #000000;
                border-radius: 0px;
                background-color: #FFFFFF;
                font-size: 13px;
                color: #000000;
            }
            QLineEdit:focus, QComboBox:focus { border: 2px solid #E50000; }
            QPushButton {
                padding: 8px 16px;
                background-color: #000000;
                color: #FFFFFF;
                border: 2px solid #000000;
                border-radius: 0px;
                font-weight: bold;
                font-size: 13px;
                text-transform: uppercase;
            }
            QPushButton:hover { background-color: #FFFFFF; color: #000000; }
            QPushButton:pressed { background-color: #E50000; color: #FFFFFF; border: 2px solid #E50000; }
            QPushButton:disabled { background-color: #CCCCCC; color: #666666; border: 2px solid #CCCCCC; }
            QPushButton#stopBtn { background-color: #E50000; color: #FFFFFF; border: 2px solid #E50000; }
            QPushButton#stopBtn:hover { background-color: #FFFFFF; color: #E50000; }
            QPlainTextEdit {
                background-color: #FFFFFF;
                color: #000000;
                border: 2px solid #000000;
                border-radius: 0px;
                padding: 8px;
                font-size: 13px;
            }
        """)

        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title_label = QLabel("SEMANTIC DESK ASSISTANT")
        title_font = QFont("Arial Black")
        title_font.setPointSize(20)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # --- Left Panel: Camera Stream ---
        camera_panel = QWidget()
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(0, 0, 10, 0)

        cam_title = QLabel("CAMERA FEED")
        cam_title.setFont(QFont("Arial Black", 14))
        camera_layout.addWidget(cam_title)

        cam_controls = QHBoxLayout()
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("ESP32 IP e.g. 192.168.137.28")
        self.ip_input.setText(self.settings.value("esp32_ip", "192.168.137.28"))
        self.ip_input.textChanged.connect(lambda text: self.settings.setValue("esp32_ip", text))
        
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(["0°", "90°", "180°", "270°"])
        self.rotation_combo.setToolTip("Rotate incoming image clockwise")
        
        self.toggle_btn = QPushButton("Start Auto Capture")
        self.toggle_btn.clicked.connect(self.toggle_capture)
        
        cam_controls.addWidget(self.ip_input)
        cam_controls.addWidget(self.rotation_combo)
        cam_controls.addWidget(self.toggle_btn)
        camera_layout.addLayout(cam_controls)
        
        test_mode_layout = QHBoxLayout()
        self.test_mode_check = QCheckBox("Test Mode (Local Images)")
        self.test_mode_check.stateChanged.connect(self.on_test_mode_changed)
        
        self.load_img1_btn = QPushButton("Load Image 1 (Past)")
        self.load_img1_btn.clicked.connect(lambda checked, s=1: self.load_local_image(step=s))
        self.load_img1_btn.setVisible(False)
        
        self.load_img2_btn = QPushButton("Load Image 2 (Present)")
        self.load_img2_btn.clicked.connect(lambda checked, s=2: self.load_local_image(step=s))
        self.load_img2_btn.setVisible(False)
        
        test_mode_layout.addWidget(self.test_mode_check)
        test_mode_layout.addWidget(self.load_img1_btn)
        test_mode_layout.addWidget(self.load_img2_btn)
        test_mode_layout.addStretch()
        camera_layout.addLayout(test_mode_layout)

        self.capture_info_label = QLabel("Ready to capture")
        self.capture_info_label.setStyleSheet("color: #636E72; font-size: 12px;")
        camera_layout.addWidget(self.capture_info_label)

        self.image_frame = QFrame()
        self.image_frame.setStyleSheet("QFrame { background-color: #FFFFFF; border: 2px solid #000000; }")
        shadow1 = QGraphicsDropShadowEffect()
        shadow1.setBlurRadius(0)
        shadow1.setColor(QColor(0, 0, 0, 255))
        shadow1.setOffset(6, 6)
        self.image_frame.setGraphicsEffect(shadow1)

        image_layout = QVBoxLayout(self.image_frame)
        self.image_label = QLabel("No image yet")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #B2BEC3; font-size: 16px;")
        image_layout.addWidget(self.image_label)
        
        camera_layout.addWidget(self.image_frame, stretch=1)
        splitter.addWidget(camera_panel)

        # --- Right Panel: Split between Events & Chat ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.setSpacing(15)
        
        right_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(right_splitter)

        # Top Right: Events Log
        events_panel = QWidget()
        events_layout = QVBoxLayout(events_panel)
        events_layout.setContentsMargins(0,0,0,0)

        events_title = QLabel("EVENTS LOG")
        events_title.setFont(QFont("Arial Black", 14))
        events_layout.addWidget(events_title)

        self.events_output = QPlainTextEdit()
        self.events_output.setReadOnly(True)
        self.events_output.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace;")
        events_layout.addWidget(self.events_output, stretch=1)
        right_splitter.addWidget(events_panel)

        # Bottom Right: AI Chat
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(0,0,0,0)

        chat_title = QLabel("AI VISION ASSISTANT")
        chat_title.setFont(QFont("Arial Black", 14))
        chat_layout.addWidget(chat_title)

        self.chat_output = QPlainTextEdit()
        self.chat_output.setReadOnly(True)
        chat_layout.addWidget(self.chat_output, stretch=1)

        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask a question about the latest capture...")
        self.chat_input.returnPressed.connect(self.send_chat)
        self.chat_input.setDisabled(True) # Disabled until model loads
        
        self.send_btn = QPushButton("Ask")
        self.send_btn.clicked.connect(self.send_chat)
        self.send_btn.setDisabled(True)
        
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.send_btn)
        chat_layout.addLayout(chat_input_layout)
        right_splitter.addWidget(chat_panel)
        
        splitter.addWidget(right_panel)
        
        # --- Far Right Panel: Objects Table ---
        objects_panel = QWidget()
        objects_layout = QVBoxLayout(objects_panel)
        objects_layout.setContentsMargins(10, 0, 0, 0)
        
        objects_title = QLabel("CURRENT OBJECTS")
        objects_title.setFont(QFont("Arial Black", 14))
        objects_layout.addWidget(objects_title)
        
        self.objects_table = QTableWidget(0, 2)
        self.objects_table.setHorizontalHeaderLabels(["Object", "Count"])
        self.objects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.objects_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.objects_table.verticalHeader().setVisible(False)
        self.objects_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.objects_table.setSelectionMode(QTableWidget.NoSelection)
        self.objects_table.setStyleSheet("""
            QTableWidget {
                background-color: #FFFFFF;
                border: 2px solid #000000;
                border-radius: 0px;
                font-size: 13px;
                color: #000000;
                gridline-color: #000000;
            }
            QHeaderView::section {
                background-color: #000000;
                color: #FFFFFF;
                padding: 6px;
                border: none;
                border-bottom: 2px solid #000000;
                border-right: 1px solid #FFFFFF;
                font-weight: bold;
            }
        """)
        objects_layout.addWidget(self.objects_table, stretch=1)
        splitter.addWidget(objects_panel)
        
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

    # --- AI Model Loading ---
    def on_models_loaded(self, model, processor):
        self.vl_model = model
        self.vl_processor = processor
        self.chat_output.appendPlainText("[System] Qwen2-VL-2B model loaded and ready! Ask away.")
        self.chat_input.setDisabled(False)
        self.send_btn.setDisabled(False)
        
    def on_models_load_error(self, error):
        self.chat_output.appendPlainText(f"[System] Failed to load model: {error}")

    # --- AI Chat Methods ---
    def send_chat(self):
        prompt = self.chat_input.text().strip()
        if not prompt: return
        
        if not self.latest_raw_image or not os.path.exists(self.latest_raw_image):
            self.chat_output.appendPlainText("[System] Wait for a camera capture first!")
            return
            
        self.chat_input.clear()
        self.chat_output.appendPlainText(f"You: {prompt}")
        self.chat_output.appendPlainText("⏳ AI is analyzing the latest image...")
        self.chat_output.moveCursor(QTextCursor.End)
        
        self.chat_input.setDisabled(True)
        self.send_btn.setDisabled(True)
        
        context_log = self.events_output.toPlainText()
        self.vqa_worker = VqaWorker(self.latest_raw_image, prompt, self.vl_model, self.vl_processor, context_log)
        self.vqa_worker.finished_processing.connect(self.on_vqa_finished)
        self.vqa_worker.error_processing.connect(self.on_vqa_error)
        self.vqa_worker.start()

    def on_vqa_finished(self, answer):
        # Remove the "⏳ AI is analyzing..." line safely
        text = self.chat_output.toPlainText()
        lines = text.split('\n')
        if lines and "⏳" in lines[-1]:
            lines.pop()
        self.chat_output.setPlainText('\n'.join(lines))
        self.chat_output.moveCursor(QTextCursor.End)
        
        self.chat_output.appendPlainText(f"AI: {answer}\n")
        self.chat_output.moveCursor(QTextCursor.End)
        self.chat_input.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.chat_input.setFocus()

    def on_vqa_error(self, err_msg):
        self.chat_output.appendPlainText(f"[System] AI Error: {err_msg}\n")
        self.chat_input.setDisabled(False)
        self.send_btn.setDisabled(False)

    def on_test_mode_changed(self, state):
        is_test_mode = self.test_mode_check.isChecked()
        if is_test_mode and self.is_running:
            self.toggle_capture()
            
        self.ip_input.setDisabled(is_test_mode)
        self.rotation_combo.setDisabled(is_test_mode)
        self.toggle_btn.setDisabled(is_test_mode)
        self.load_img1_btn.setVisible(is_test_mode)
        self.load_img2_btn.setVisible(is_test_mode)
        
        if is_test_mode:
            self.capture_info_label.setText("Test Mode active. Load Image 1 to begin.")

    def load_local_image(self, step=1, file_path=None):
        if not isinstance(file_path, str):
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"Select Image {step}", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
        if file_path and os.path.exists(file_path):
            if step == 1:
                self.previous_objects = {}
                self.events_output.clear()
                self.objects_table.setRowCount(0)
                
            self.settings.setValue(f"last_test_image_{step}", file_path)
            self.latest_raw_image = file_path
            
            if self.vl_model:
                if not hasattr(self, 'worker') or not self.worker.isRunning():
                    self.capture_info_label.setText(f"Running VL Detection on {os.path.basename(file_path)}...")
                    self.worker = VlDetectionWorker(file_path, self.annotated_dir, self.vl_model, self.vl_processor)
                    self.worker.finished_processing.connect(self.on_detection_finished)
                    self.worker.error_processing.connect(self.on_detection_error)
                    self.worker.start()
            else:
                self.capture_info_label.setText(f"Loaded local image: {os.path.basename(file_path)}")
                self.show_image_from_path(file_path)

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
        if not self.is_running: return
        request = QNetworkRequest(QUrl(self.url))
        self.network_manager.get(request)

    def on_image_downloaded(self, reply):
        if not self.is_running:
            reply.deleteLater()
            return
            
        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = reply.readAll()
            
            try:
                nparr = np.frombuffer(data.data(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    rotation = self.rotation_combo.currentText()
                    if rotation == "90°": img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == "180°": img = cv2.rotate(img, cv2.ROTATE_180)
                    elif rotation == "270°": img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    raw_filename = os.path.join(self.raw_dir, f"capture_{timestamp}.jpg")
                    cv2.imwrite(raw_filename, img)
                    
                    self.latest_raw_image = raw_filename
                    
                    if self.vl_model:
                        if not hasattr(self, 'worker') or not self.worker.isRunning():
                            self.capture_info_label.setText(f"Running VL Detection on {timestamp}...")
                            self.worker = VlDetectionWorker(raw_filename, self.annotated_dir, self.vl_model, self.vl_processor)
                            self.worker.finished_processing.connect(self.on_detection_finished)
                            self.worker.error_processing.connect(self.on_detection_error)
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

    def on_detection_finished(self, annotated_path, current_objects):
        self.capture_info_label.setText(f"Last capture: {time.strftime('%H:%M:%S')} - Saved raw & annotated")
        self.show_image_from_path(annotated_path)
        
        all_classes = set(current_objects.keys()).union(set(self.previous_objects.keys()))
        events = []
        
        for cls in all_classes:
            current_count = current_objects.get(cls, 0)
            prev_count = self.previous_objects.get(cls, 0)
            
            diff = current_count - prev_count
            if diff > 0: events.append(f"Detected: {diff}x {cls}")
            elif diff < 0: events.append(f"Removed: {abs(diff)}x {cls}")
                
        if events:
            for event in events: self.log_event(event)
                
        self.previous_objects = current_objects
        
        # Update the Objects Table
        self.objects_table.setRowCount(0)
        for obj_name, count in current_objects.items():
            row_idx = self.objects_table.rowCount()
            self.objects_table.insertRow(row_idx)
            
            obj_item = QTableWidgetItem(str(obj_name))
            self.objects_table.setItem(row_idx, 0, obj_item)
            
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.objects_table.setItem(row_idx, 1, count_item)

    def on_detection_error(self, err_msg):
        self.capture_info_label.setText(f"Detection Error: {err_msg}")
        self.log_event(f"Detection Error: {err_msg}")

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
        
        self.events_output.appendPlainText(f"{short_time} {message}")
        self.events_output.moveCursor(QTextCursor.End)
        
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
