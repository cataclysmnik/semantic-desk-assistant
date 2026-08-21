from PySide6.QtWidgets import QApplication, QPushButton

def test_lambda():
    def callback(step):
        print(f"Callback called with step={step}")
    
    app = QApplication([])
    btn = QPushButton("Test")
    # Simulate how it's connected
    btn.clicked.connect(lambda checked, s=1: callback(s))
    btn.click()

test_lambda()
