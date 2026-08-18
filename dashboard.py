import sys
import os
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QGraphicsDropShadowEffect, QFrame, QComboBox, 
                               QPlainTextEdit, QSplitter)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QPixmap, QFont, QColor, QTextCursor
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PySide6.QtSerialPort import QSerialPort, QSerialPortInfo

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 Cam & Serial Dashboard")
        self.setMinimumSize(1000, 700)
        
        # --- Camera State ---
        self.is_running = False
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.on_image_downloaded)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fetch_image)
        # Fetch every 10 seconds for auto capture
        self.timer.setInterval(10000)

        # Folder to save images
        self.save_dir = "esp32_captures"

        # --- Serial State ---
        self.serial = QSerialPort(self)
        self.serial.readyRead.connect(self.read_serial_data)

        self.setup_ui()
        self.populate_serial_ports()

    def setup_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Apply modern light mode stylesheet
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
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)

        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Header Title
        title_label = QLabel("ESP32 Control Center")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # Splitter to separate camera view and serial monitor
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # --- Left Panel: Camera Stream ---
        camera_panel = QWidget()
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(0, 0, 10, 0)

        cam_title = QLabel("Auto Capture (10s)")
        cam_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        camera_layout.addWidget(cam_title)

        cam_controls = QHBoxLayout()
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("ESP32 IP e.g. 192.168.137.28")
        self.ip_input.setText("192.168.137.28")
        
        self.toggle_btn = QPushButton("Start Auto Capture")
        self.toggle_btn.clicked.connect(self.toggle_capture)
        
        cam_controls.addWidget(self.ip_input)
        cam_controls.addWidget(self.toggle_btn)
        camera_layout.addLayout(cam_controls)

        # Subtitle to show latest capture time
        self.capture_info_label = QLabel("Ready to capture")
        self.capture_info_label.setStyleSheet("color: #636E72; font-size: 12px;")
        camera_layout.addWidget(self.capture_info_label)

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

        # --- Right Panel: Serial Monitor ---
        serial_panel = QWidget()
        serial_layout = QVBoxLayout(serial_panel)
        serial_layout.setContentsMargins(10, 0, 0, 0)

        serial_title = QLabel("Serial Monitor")
        serial_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        serial_layout.addWidget(serial_title)

        serial_controls = QHBoxLayout()
        self.port_combo = QComboBox()
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "115200", "460800"])
        self.baud_combo.setCurrentText("115200")

        self.refresh_ports_btn = QPushButton("Refresh")
        self.refresh_ports_btn.clicked.connect(self.populate_serial_ports)

        self.serial_btn = QPushButton("Connect")
        self.serial_btn.clicked.connect(self.toggle_serial)

        serial_controls.addWidget(self.port_combo, stretch=2)
        serial_controls.addWidget(self.baud_combo, stretch=1)
        serial_controls.addWidget(self.refresh_ports_btn)
        serial_controls.addWidget(self.serial_btn)
        serial_layout.addLayout(serial_controls)

        self.serial_output = QPlainTextEdit()
        self.serial_output.setReadOnly(True)
        
        shadow2 = QGraphicsDropShadowEffect()
        shadow2.setBlurRadius(15)
        shadow2.setColor(QColor(0, 0, 0, 15))
        shadow2.setOffset(0, 3)
        self.serial_output.setGraphicsEffect(shadow2)

        serial_layout.addWidget(self.serial_output, stretch=1)
        
        clear_btn = QPushButton("Clear Output")
        clear_btn.clicked.connect(self.serial_output.clear)
        serial_layout.addWidget(clear_btn)

        splitter.addWidget(serial_panel)
        
        # Set stretch factors (Camera 60%, Serial 40%)
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
            # Trigger first fetch immediately, then start timer for subsequent fetches
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
            
            # Save the image locally
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            os.makedirs(self.save_dir, exist_ok=True)
            filename = os.path.join(self.save_dir, f"capture_{timestamp}.jpg")
            
            try:
                with open(filename, "wb") as f:
                    f.write(data)
                self.capture_info_label.setText(f"Last capture: {time.strftime('%H:%M:%S')} - Saved to {self.save_dir}/")
            except Exception as e:
                self.capture_info_label.setText(f"Failed to save image: {e}")

            # Display the image in the GUI
            pixmap = QPixmap()
            if pixmap.loadFromData(data):
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("Error decoding image")
        else:
            self.image_label.setText(f"Connection error: {reply.errorString()}")
            self.capture_info_label.setText("Retrying in 10s...")
            
        reply.deleteLater()

    # --- Serial Monitor Methods ---
    def populate_serial_ports(self):
        self.port_combo.clear()
        ports = QSerialPortInfo.availablePorts()
        for port in ports:
            self.port_combo.addItem(f"{port.portName()} - {port.description()}", port.portName())

    def toggle_serial(self):
        if not self.serial.isOpen():
            port_name = self.port_combo.currentData()
            baud_rate = int(self.baud_combo.currentText())
            
            if not port_name:
                self.serial_output.appendPlainText("No port selected.")
                return

            self.serial.setPortName(port_name)
            self.serial.setBaudRate(baud_rate)
            
            if self.serial.open(QSerialPort.ReadWrite):
                self.serial_btn.setText("Disconnect")
                self.serial_btn.setObjectName("stopBtn")
                self.serial_btn.style().unpolish(self.serial_btn)
                self.serial_btn.style().polish(self.serial_btn)
                self.serial_output.appendPlainText(f"--- Connected to {port_name} at {baud_rate} baud ---")
            else:
                self.serial_output.appendPlainText(f"--- Failed to connect to {port_name}: {self.serial.errorString()} ---")
        else:
            self.serial.close()
            self.serial_btn.setText("Connect")
            self.serial_btn.setObjectName("")
            self.serial_btn.style().unpolish(self.serial_btn)
            self.serial_btn.style().polish(self.serial_btn)
            self.serial_output.appendPlainText("--- Disconnected ---")

    def read_serial_data(self):
        if self.serial.canReadLine():
            data = self.serial.readAll().data().decode('utf-8', errors='replace').strip()
            if data:
                self.serial_output.appendPlainText(data)
                # Auto-scroll to bottom
                self.serial_output.moveCursor(QTextCursor.End)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Dashboard()
    window.show()
    sys.exit(app.exec())
