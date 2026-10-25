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
from PySide6.QtGui import QPixmap, QFont, QColor, QTextCursor, QIcon
from PySide6.QtWidgets import QSystemTrayIcon, QMenu
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

try:
    from web_server import start_web_server_thread, update_server_state, get_local_ip
except ImportError:
    start_web_server_thread = lambda port=8080: (None, "127.0.0.1", port)
    update_server_state = lambda data: None
    get_local_ip = lambda: "127.0.0.1"

try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True)
except ImportError:
    print("Please install pyqtgraph (pip install pyqtgraph) for interactive timeline graphs.")
    pg = None



try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Please install transformers and torch (pip install transformers torch pillow) for AI Chat.")
    AutoModelForCausalLM = None
    AutoTokenizer = None


class ModelLoader(QThread):
    models_loaded = Signal(object, object)
    error_loading = Signal(str)
    
    def run(self):
        try:
            if not AutoModelForCausalLM:
                raise Exception("Transformers library not found.")
                
            model_id = "./moondream2"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
                
            self.models_loaded.emit(model, tokenizer)
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            self.error_loading.emit(f"{str(e)}\n\nTraceback:\n{full_traceback}")


class VlDetectionWorker(QThread):
    finished_processing = Signal(str, dict)
    error_processing = Signal(str)

    def __init__(self, raw_path, annotated_dir, model, tokenizer):
        super().__init__()
        self.raw_path = raw_path
        self.annotated_dir = annotated_dir
        self.model = model
        self.tokenizer = tokenizer

    def run(self):
        try:
            if not self.model:
                raise Exception("VL model is not loaded.")
            
            image = Image.open(self.raw_path).convert("RGB")

            prompt = "What objects are on the desk?"

            enc_image = self.model.encode_image(image)
            answer = self.model.answer_question(
                enc_image, prompt, tokenizer=self.tokenizer,
                repetition_penalty=1.15,
            )
            
            import json
            import shutil
            import re
            
            # Moondream2 responds with concise descriptions like: "On the desk, there is a laptop, a cell phone, and a book."
            detected_counts = {}
            ans_lower = answer.lower()
            
            # Map canonical items based on what moondream2 actually recognized
            item_rules = [
                ('laptop', ['open laptop', 'laptop']),
                ('closed laptop', ['closed laptop']),
                ('computer', ['computer', 'monitor', 'pc']),
                ('keyboard', ['keyboard']),
                ('mouse', ['mouse']),
                ('phone with screen on', ['cell phone', 'phone', 'mobile']),
                ('book', ['book']),
                ('notebook', ['notebook', 'notepad']),
                ('paper', ['paper', 'document', 'sheet']),
                ('pen', ['pen', 'pencil']),
                ('mug', ['mug', 'cup', 'coffee']),
                ('bottle', ['bottle', 'water bottle']),
                ('bowl', ['bowl', 'plate']),
            ]
            
            for canon_name, synonyms in item_rules:
                for syn in synonyms:
                    if re.search(rf'\b{re.escape(syn)}\b', ans_lower):
                        # Distinguish open vs closed laptop if possible
                        if canon_name == 'laptop':
                            c_name = 'closed laptop' if 'closed' in ans_lower else 'open laptop'
                        elif canon_name == 'phone with screen on':
                            c_name = 'phone with screen on' if ('screen on' in ans_lower or 'display' in ans_lower or 'using' in ans_lower) else 'phone with screen off'
                        else:
                            c_name = canon_name
                        detected_counts[c_name] = detected_counts.get(c_name, 0) + 1
                        break
                
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

    def __init__(self, raw_path, prompt, model, tokenizer, context_log=""):
        super().__init__()
        self.raw_path = raw_path
        self.prompt = prompt
        self.model = model
        self.tokenizer = tokenizer
        self.context_log = context_log

    def run(self):
        try:
            image = Image.open(self.raw_path).convert("RGB")

            # Build full question with context for moondream2
            full_question = self.prompt
            if self.context_log.strip():
                full_question = (
                    f"Context — here is a log of recent object detection events "
                    f"(Past -> Present):\n{self.context_log}\n\n"
                    f"Question: {self.prompt}"
                )

            enc_image = self.model.encode_image(image)
            answer = self.model.answer_question(
                enc_image, full_question, tokenizer=self.tokenizer,
            )

            self.finished_processing.emit(answer)
        except Exception as e:
            self.error_processing.emit(str(e))


if pg:
    class TimeAxisItem(pg.AxisItem):
        """X-axis that safely displays Unix timestamps as local HH:MM:SS."""
        def tickStrings(self, values, scale, spacing):
            try:
                strings = []
                for v in values:
                    try:
                        val = float(v)
                        # Windows time.localtime expects 0 <= timestamp <= 32535215999 (year 3000).
                        # Guard against negative, NaN, Inf, or dummy initialization coordinates (< 86400).
                        if not np.isfinite(val) or val < 86400 or val > 32535215999:
                            strings.append("")
                        else:
                            strings.append(time.strftime('%H:%M:%S', time.localtime(val)))
                    except Exception:
                        strings.append("")
                return strings
            except Exception:
                return ["" for _ in values] if values else []
else:
    TimeAxisItem = None


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
        self.current_session = "IDLE"
        self.time_history = []
        self.session_history = []
        self.image_history = []
        self.object_history = {}
        self.session_map = {"IDLE": 0, "BREAK": 1, "STUDY": 2, "WORK": 3}

        # Productivity tracking
        self.session_durations = {"IDLE": 0.0, "BREAK": 0.0, "STUDY": 0.0, "WORK": 0.0}
        self._session_start_time = None   # wall-clock time when current session began
        self.past_sessions = []
        
        self.vl_model = None
        self.vl_tokenizer = None
        self.latest_raw_image = None
        
        self.video_cap = None
        self.video_msec = 0
        
        self.is_running = False
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.on_image_downloaded)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fetch_image)
        self.timer.setInterval(10000)

        self.setup_ui()
        self._setup_tray()
        
        # Start background Mobile Monitor Web Server
        self.web_server, self.local_ip, self.web_port = start_web_server_thread(8080)
        if hasattr(self, 'mobile_url_label') and self.local_ip:
            self.mobile_url_label.setText(f"PHONE MONITOR: http://{self.local_ip}:{self.web_port}")
        
        # Start loading Moondream2 in the background
        if AutoModelForCausalLM:
            self.chat_output.appendPlainText("[System] Loading Moondream2 model in the background... This may take a moment.")
            self.model_loader = ModelLoader()
            self.model_loader.models_loaded.connect(self.on_models_loaded)
            self.model_loader.error_loading.connect(self.on_models_load_error)
            self.model_loader.start()
        else:
            self.chat_output.appendPlainText("[System] Please install transformers to use AI chat.")
            self.chat_input.setDisabled(True)
            self.send_btn.setDisabled(True)

    SESSION_ICONS = {
        "IDLE":  "",
        "BREAK": "",
        "STUDY": "",
        "WORK":  "",
    }
    SESSION_MSGS = {
        "IDLE":  "No activity detected.",
        "BREAK": "Taking a break.",
        "STUDY": "Study session started.",
        "WORK":  "Work session started.",
    }

    def _setup_tray(self):
        """Create a system-tray icon used to deliver session-change notifications."""
        self._tray = QSystemTrayIcon(self)
        # Use a built-in Qt icon as a fallback so we always have something visible.
        self._tray.setIcon(self.style().standardIcon(
            self.style().StandardPixmap.SP_ComputerIcon
        ))
        self._tray.setToolTip("Semantic Desk Assistant")

        tray_menu = QMenu()
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.instance().quit)
        self._tray.setContextMenu(tray_menu)

        self._tray.show()

    def _notify_session(self, old_session: str, new_session: str):
        """Fire a desktop notification for a session transition."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        msg = self.SESSION_MSGS.get(new_session, "")
        title = f"Session: {new_session}"
        body  = f"{old_session} -> {new_session}  |  {msg}"
        self._tray.showMessage(
            title,
            body,
            QSystemTrayIcon.MessageIcon.Information,
            4000,   # display duration in ms
        )

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

        title_layout = QHBoxLayout()
        title_label = QLabel("SEMANTIC DESK ASSISTANT")
        title_font = QFont("Arial Black")
        title_font.setPointSize(20)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)

        self.mobile_url_label = QLabel(f"PHONE MONITOR: http://{get_local_ip()}:8080")
        self.mobile_url_label.setFont(QFont("Consolas", 12, QFont.Bold))
        self.mobile_url_label.setStyleSheet("color: #E50000; padding: 4px 8px; border: 2px solid #000000; background-color: #FFFFFF;")
        self.mobile_url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        title_layout.addWidget(self.mobile_url_label, alignment=Qt.AlignRight | Qt.AlignVCenter)
        main_layout.addLayout(title_layout)

        self.main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.main_splitter, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(splitter)

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
        
        self.video_mode_check = QCheckBox("Video Mode")
        self.video_mode_check.stateChanged.connect(self.on_video_mode_changed)
        
        self.load_vid_btn = QPushButton("Load Video")
        self.load_vid_btn.clicked.connect(self.load_local_video)
        self.load_vid_btn.setVisible(False)
        
        test_mode_layout.addWidget(self.test_mode_check)
        test_mode_layout.addWidget(self.video_mode_check)
        test_mode_layout.addWidget(self.load_img1_btn)
        test_mode_layout.addWidget(self.load_img2_btn)
        test_mode_layout.addWidget(self.load_vid_btn)
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
        
        from PySide6.QtWidgets import QSizePolicy
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMinimumSize(320, 240)
        
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

        events_header = QHBoxLayout()
        events_title = QLabel("EVENTS LOG")
        events_title.setFont(QFont("Arial Black", 14))
        events_header.addWidget(events_title)
        
        self.session_label = QLabel("SESSION: IDLE")
        self.session_label.setFont(QFont("Arial Black", 14))
        self.session_label.setStyleSheet("color: #E50000;")
        events_header.addWidget(self.session_label, alignment=Qt.AlignRight)
        
        events_layout.addLayout(events_header)

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
        
        # --- Bottom Panel: Graphs + Productivity Score ---
        self.graph_panel = QWidget()
        graph_layout = QHBoxLayout(self.graph_panel)
        graph_layout.setSpacing(15)

        if pg:
            font = QFont("Arial Black", 10)

            self.session_plot = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
            self.session_plot.setBackground('#FFFFFF')
            self.session_plot.showGrid(x=True, y=True, alpha=0.3)
            self.session_plot.getPlotItem().setTitle("SESSION TIMELINE", color='#000000', size='12pt', bold=True)

            left_axis = self.session_plot.getAxis('left')
            left_axis.setTicks([[(0, 'IDLE'), (1, 'BREAK'), (2, 'STUDY'), (3, 'WORK')]])
            left_axis.setPen(pg.mkPen(color='#000000', width=2))
            left_axis.setTextPen(pg.mkPen(color='#000000'))
            left_axis.tickFont = font

            bottom_axis_s = self.session_plot.getAxis('bottom')
            bottom_axis_s.setPen(pg.mkPen(color='#000000', width=2))
            bottom_axis_s.setTextPen(pg.mkPen(color='#000000'))
            bottom_axis_s.tickFont = font

            self.scrub_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#000000', width=2, style=Qt.DashLine))
            self.scrub_line.sigDragged.connect(self.on_scrub)
            self.session_plot.addItem(self.scrub_line)

            graph_layout.addWidget(self.session_plot, stretch=4)
        else:
            graph_layout.addWidget(QLabel("Please run 'pip install pyqtgraph' in your virtual environment to view timelines."), stretch=4)

        # --- Productivity Score Widget ---
        score_frame = QFrame()
        score_frame.setStyleSheet(
            "QFrame { background-color: #FFFFFF; border: 2px solid #000000; }"
        )
        shadow_s = QGraphicsDropShadowEffect()
        shadow_s.setBlurRadius(0)
        shadow_s.setColor(QColor(0, 0, 0, 255))
        shadow_s.setOffset(6, 6)
        score_frame.setGraphicsEffect(shadow_s)

        score_inner = QVBoxLayout(score_frame)
        score_inner.setContentsMargins(14, 14, 14, 14)
        score_inner.setSpacing(8)

        score_heading = QLabel("PRODUCTIVITY")
        score_heading.setFont(QFont("Arial Black", 11))
        score_heading.setAlignment(Qt.AlignCenter)
        score_inner.addWidget(score_heading)

        self.score_value_label = QLabel("—")
        self.score_value_label.setFont(QFont("Arial Black", 36))
        self.score_value_label.setAlignment(Qt.AlignCenter)
        score_inner.addWidget(self.score_value_label)

        self.score_grade_label = QLabel("No data yet")
        self.score_grade_label.setFont(QFont("Arial Black", 13))
        self.score_grade_label.setAlignment(Qt.AlignCenter)
        score_inner.addWidget(self.score_grade_label)

        # Breakdown rows
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #000000;")
        score_inner.addWidget(sep)

        self.score_work_label  = QLabel("WORK   0m")
        self.score_study_label = QLabel("STUDY  0m")
        self.score_break_label = QLabel("BREAK  0m")
        self.score_idle_label  = QLabel("IDLE   0m")
        for lbl in (self.score_work_label, self.score_study_label,
                    self.score_break_label, self.score_idle_label):
            lbl.setFont(QFont("Consolas", 11))
            lbl.setAlignment(Qt.AlignLeft)
            score_inner.addWidget(lbl)

        score_inner.addStretch()
        graph_layout.addWidget(score_frame, stretch=1)

        self.main_splitter.addWidget(self.graph_panel)
        self.main_splitter.setStretchFactor(0, 7)
        self.main_splitter.setStretchFactor(1, 3)

    # --- AI Model Loading ---
    def on_models_loaded(self, model, tokenizer):
        self.vl_model = model
        self.vl_tokenizer = tokenizer
        self.chat_output.appendPlainText("[System] Moondream2 model loaded and ready! Ask away.")
        self.chat_input.setDisabled(False)
        self.send_btn.setDisabled(False)
        
    def on_models_load_error(self, error):
        self.chat_output.appendPlainText(f"[System] Failed to load model: {error}")

    def on_scrub(self):
        if not self.time_history: return
        val = self.scrub_line.value()
        idx = (np.abs(np.array(self.time_history) - val)).argmin()
        img_path = self.image_history[idx]
        if os.path.exists(img_path):
            self.show_image_from_path(img_path)
            session_at_idx = list(self.session_map.keys())[list(self.session_map.values()).index(self.session_history[idx])]
            try:
                timestamp = time.strftime('%H:%M:%S', time.localtime(self.time_history[idx]))
            except Exception:
                timestamp = "Unknown"
            self.capture_info_label.setText(f"Viewing history at {timestamp} - Session: {session_at_idx}")

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
        self.vqa_worker = VqaWorker(self.latest_raw_image, prompt, self.vl_model, self.vl_tokenizer, context_log)
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
            
        self.ip_input.setDisabled(is_test_mode or hasattr(self, 'video_mode_check') and self.video_mode_check.isChecked())
        self.rotation_combo.setDisabled(is_test_mode)
        self.toggle_btn.setDisabled(is_test_mode)
        self.load_img1_btn.setVisible(is_test_mode)
        self.load_img2_btn.setVisible(is_test_mode)
        self.video_mode_check.setDisabled(is_test_mode)
        
        if is_test_mode:
            self.capture_info_label.setText("Test Mode active. Load Image 1 to begin.")

    def on_video_mode_changed(self, state):
        is_video_mode = self.video_mode_check.isChecked()
        if is_video_mode and self.is_running:
            self.toggle_capture()
            
        self.ip_input.setDisabled(is_video_mode or self.test_mode_check.isChecked())
        self.load_vid_btn.setVisible(is_video_mode)
        self.test_mode_check.setDisabled(is_video_mode)
        
        if is_video_mode:
            self.capture_info_label.setText("Video Mode active. Load a video and Start Auto Capture.")

    def load_local_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if file_path and os.path.exists(file_path):
            if self.video_cap:
                self.video_cap.release()
            self.video_cap = cv2.VideoCapture(file_path)
            self.video_msec = 0
            
            # Reset history
            self.previous_objects = {}
            self.events_output.clear()
            self.objects_table.setRowCount(0)
            
            self.capture_info_label.setText(f"Loaded video: {os.path.basename(file_path)}")

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
                    self.worker = VlDetectionWorker(file_path, self.annotated_dir, self.vl_model, self.vl_tokenizer)
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
        
        if hasattr(self, 'video_mode_check') and self.video_mode_check.isChecked():
            if self.video_cap and self.video_cap.isOpened():
                self.video_cap.set(cv2.CAP_PROP_POS_MSEC, self.video_msec)
                ret, frame = self.video_cap.read()
                if ret:
                    self.process_frame(frame)
                    self.video_msec += self.timer.interval()
                else:
                    self.capture_info_label.setText("End of video reached.")
                    self.toggle_capture()
        else:
            request = QNetworkRequest(QUrl(self.url))
            self.network_manager.get(request)

    def process_frame(self, img):
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
                    self.worker = VlDetectionWorker(raw_filename, self.annotated_dir, self.vl_model, self.vl_tokenizer)
                    self.worker.finished_processing.connect(self.on_detection_finished)
                    self.worker.error_processing.connect(self.on_detection_error)
                    self.worker.start()
            else:
                self.capture_info_label.setText(f"Last capture: {time.strftime('%H:%M:%S')} - Saved to {self.raw_dir}")
                self.show_image_from_path(raw_filename)
        else:
            self.capture_info_label.setText("Failed to decode image data.")

    def on_image_downloaded(self, reply):
        if not self.is_running:
            reply.deleteLater()
            return
            
        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = reply.readAll()
            
            try:
                nparr = np.frombuffer(data.data(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.process_frame(img)
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
        
        # --- Session Detection ---
        # Only consider objects that actually have a count > 0
        obj_keys = [k.lower() for k, count in current_objects.items() if count > 0]
        
        has_work = any((("laptop" in k and "closed" not in k) or "computer" in k or "keyboard" in k) for k in obj_keys)
        has_study = any(x in k for k in obj_keys for x in ["book", "notebook", "notepad", "paper"])
        
        if not obj_keys:
            new_session = "IDLE"
        elif has_work:
            new_session = "WORK"
        elif has_study:
            new_session = "STUDY"
        else:
            new_session = "BREAK"
            
        if new_session != self.current_session:
            old_session = self.current_session

            # Accumulate time spent in the outgoing session
            now_time = time.time()
            if self._session_start_time is not None:
                elapsed = now_time - self._session_start_time
                self.session_durations[old_session] = \
                    self.session_durations.get(old_session, 0.0) + elapsed
                
                # Record to past_sessions history
                start_hms = time.strftime("%H:%M:%S", time.localtime(self._session_start_time))
                end_hms = time.strftime("%H:%M:%S", time.localtime(now_time))
                self.past_sessions.append({
                    "session": old_session,
                    "start_time": start_hms,
                    "end_time": end_hms,
                    "duration_seconds": round(elapsed),
                    "date": time.strftime("%Y-%m-%d")
                })
            self._session_start_time = now_time

            if old_session != "IDLE":
                self.log_event(f"Session Ended: {old_session}")
            if new_session != "IDLE":
                self.log_event(f"Session Started: {new_session}")

            self.current_session = new_session
            self.session_label.setText(f"SESSION: {self.current_session}")
            self._notify_session(old_session, new_session)
        else:
            # Same session: still tick the clock so first-ever session is counted
            if self._session_start_time is None:
                self._session_start_time = time.time()

        self._update_score()
        self._sync_web_state(annotated_path, current_objects)
            
        if pg:
            current_time = time.time()
            self.time_history.append(current_time)
            self.session_history.append(self.session_map.get(self.current_session, 0))
            self.image_history.append(annotated_path)
            
            for obj in set(list(current_objects.keys()) + list(self.object_history.keys())):
                if obj not in self.object_history:
                    # Pad previous timestamps with np.nan so they don't slope from 0
                    self.object_history[obj] = [np.nan] * (len(self.time_history) - 1)
                
                count = current_objects.get(obj, 0)
                # Store np.nan instead of 0 so the line completely vanishes instead of sliding to 0
                self.object_history[obj].append(count if count > 0 else np.nan)
                
            # Plot
            self.session_plot.clear()
            
            colors = {
                "IDLE": (149, 165, 166),   # Gray
                "BREAK": (241, 196, 15),   # Yellow
                "STUDY": (52, 152, 219),   # Blue
                "WORK": (231, 76, 60)      # Red
            }
            
            # Draw thin dashed line for transitions
            self.session_plot.plot(self.time_history, self.session_history, pen=pg.mkPen('#CCCCCC', width=2, style=Qt.DashLine))
            
            for sess_name, sess_val in self.session_map.items():
                y_vals = [val if val == sess_val else np.nan for val in self.session_history]
                if any(not np.isnan(v) for v in y_vals):
                    self.session_plot.plot(
                        self.time_history, y_vals, 
                        connect='finite', 
                        pen=pg.mkPen(colors[sess_name], width=4),
                        symbol='o',
                        symbolSize=8,
                        symbolBrush=colors[sess_name],
                        symbolPen='#FFFFFF'
                    )
                    
            self.session_plot.addItem(self.scrub_line)
            # Advance scrub line to current time if it's roughly at the end
            if len(self.time_history) <= 1 or self.scrub_line.value() >= self.time_history[-2] - 1:
                self.scrub_line.setValue(current_time)
        
        # Update the Objects Table
        self.objects_table.setRowCount(0)
        for obj_name, count in current_objects.items():
            if count > 0:
                row_idx = self.objects_table.rowCount()
                self.objects_table.insertRow(row_idx)
                
                obj_item = QTableWidgetItem(str(obj_name))
                self.objects_table.setItem(row_idx, 0, obj_item)
                
                count_item = QTableWidgetItem(str(count))
                count_item.setTextAlignment(Qt.AlignCenter)
                self.objects_table.setItem(row_idx, 1, count_item)

    # --- Productivity Score ---
    def _update_score(self):
        """Recompute the productivity score from accumulated session durations and refresh labels."""
        # Add the currently-running session's live time without committing it
        live_durations = dict(self.session_durations)
        if self._session_start_time is not None:
            live_elapsed = time.time() - self._session_start_time
            live_durations[self.current_session] = \
                live_durations.get(self.current_session, 0.0) + live_elapsed

        focused = live_durations.get("WORK", 0.0) + live_durations.get("STUDY", 0.0)
        total   = sum(live_durations.values())

        if total < 1:
            return  # not enough data yet

        score = int(round(focused / total * 100))

        # Colour + grade
        if score >= 75:
            color, grade = "#27AE60", "Excellent"
        elif score >= 50:
            color, grade = "#2980B9", "Good"
        elif score >= 25:
            color, grade = "#E67E22", "Fair"
        else:
            color, grade = "#E50000", "Low"

        self.score_value_label.setText(f"{score}%")
        self.score_value_label.setStyleSheet(f"color: {color};")
        self.score_grade_label.setText(grade)
        self.score_grade_label.setStyleSheet(f"color: {color};")

        def fmt(secs):
            m = int(secs // 60)
            s = int(secs % 60)
            return f"{m}m {s:02d}s"

        self.score_work_label.setText( f"WORK   {fmt(live_durations.get('WORK',  0))}")
        self.score_study_label.setText(f"STUDY  {fmt(live_durations.get('STUDY', 0))}")
        self.score_break_label.setText(f"BREAK  {fmt(live_durations.get('BREAK', 0))}")
        self.score_idle_label.setText( f"IDLE   {fmt(live_durations.get('IDLE',  0))}")

    def _sync_web_state(self, annotated_path, current_objects):
        """Broadcasts current desk state to the local web monitor for mobile viewing."""
        try:
            live_durations = dict(self.session_durations)
            if self._session_start_time is not None:
                live_durations[self.current_session] = \
                    live_durations.get(self.current_session, 0.0) + (time.time() - self._session_start_time)
            
            focused = live_durations.get("WORK", 0.0) + live_durations.get("STUDY", 0.0)
            total = sum(live_durations.values())
            score = int(round(focused / total * 100)) if total >= 1 else 0
            
            if score >= 75: grade = "Excellent"
            elif score >= 50: grade = "Good"
            elif score >= 25: grade = "Fair"
            else: grade = "Low"
            
            raw_events = self.events_output.toPlainText().strip().split('\n')
            recent_events = [e.strip() for e in raw_events if e.strip()][-20:]
            
            update_server_state({
                "session": self.current_session,
                "score": score,
                "grade": grade,
                "durations": {k: round(v) for k, v in live_durations.items()},
                "objects": {k: count for k, count in current_objects.items() if count > 0},
                "latest_image": os.path.basename(annotated_path) if annotated_path else "",
                "last_capture_time": time.strftime("%H:%M:%S"),
                "recent_events": recent_events,
                "past_sessions": self.past_sessions
            })
        except Exception as e:
            print(f"[Web State Sync] Warning: {e}")

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
