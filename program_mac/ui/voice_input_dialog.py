"""
Voice and text input dialog
Supports both voice and text input
"""
try:
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QLineEdit
    from PyQt5.QtCore import Qt, pyqtSignal, QObject
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    QDialog = None
    QVBoxLayout = None
    QHBoxLayout = None
    QLabel = None
    QPushButton = None
    QTextEdit = None
    QLineEdit = None
    QObject = None
    pyqtSignal = None


if PYQT5_AVAILABLE and QDialog is not None and QObject is not None:
    class VoiceInputDialog(QDialog):
        """Voice and text input dialog"""
        
        status_updated = pyqtSignal(str)
        result_updated = pyqtSignal(str)
        button_state_updated = pyqtSignal(bool, bool)
        text_input_submitted = pyqtSignal(str)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Input Interface - Text/Voice")
            self.setMinimumWidth(500)
            self.setMinimumHeight(350)
            
            layout = QVBoxLayout()
            
            title_label = QLabel("Input Method Selection")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            layout.addWidget(title_label)
            
            text_section_label = QLabel("📝 Text Input")
            text_section_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
            layout.addWidget(text_section_label)
            
            text_input_layout = QHBoxLayout()
            self.text_input = QLineEdit()
            self.text_input.setPlaceholderText("Enter text here, then click send...")
            self.text_input.returnPressed.connect(self.on_text_send)
            text_input_layout.addWidget(self.text_input)
            
            self.send_button = QPushButton("Send")
            self.send_button.setMinimumWidth(80)
            self.send_button.clicked.connect(self.on_text_send)
            text_input_layout.addWidget(self.send_button)
            layout.addLayout(text_input_layout)
            
            separator = QLabel("─" * 50)
            separator.setAlignment(Qt.AlignCenter)
            separator.setStyleSheet("color: #ccc; margin: 10px 0px;")
            layout.addWidget(separator)
            
            voice_section_label = QLabel("🎤 Voice Input")
            voice_section_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
            layout.addWidget(voice_section_label)
            
            info_label = QLabel("Click 'Start Recording' button, then speak.\nClick 'Stop Recording' when done.")
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)
            
            self.status_label = QLabel("Status: Waiting to start...")
            self.status_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.status_label)
            
            self.result_text = QTextEdit()
            self.result_text.setReadOnly(True)
            self.result_text.setMaximumHeight(100)
            self.result_text.setPlaceholderText("Voice recognition results will appear here...")
            layout.addWidget(self.result_text)
            
            voice_button_layout = QHBoxLayout()
            self.start_button = QPushButton("Start Recording")
            self.stop_button = QPushButton("Stop Recording")
            self.stop_button.setEnabled(False)
            voice_button_layout.addWidget(self.start_button)
            voice_button_layout.addWidget(self.stop_button)
            layout.addLayout(voice_button_layout)
            
            self.setLayout(layout)
            
            self.status_updated.connect(self._update_status)
            self.result_updated.connect(self._update_result)
            self.button_state_updated.connect(self._update_button_state)
            
            self.start_button.clicked.connect(self.on_start)
            self.stop_button.clicked.connect(self.on_stop)
            
            self.is_recording = False
        
        def _update_status(self, text):
            """Update status label (called in main thread)"""
            self.status_label.setText(text)
        
        def _update_result(self, text):
            """Update result text (called in main thread)"""
            self.result_text.setText(text)
        
        def _update_button_state(self, start_enabled, stop_enabled):
            """Update button state (called in main thread)"""
            self.start_button.setEnabled(start_enabled)
            self.stop_button.setEnabled(stop_enabled)
        
        def update_status_safe(self, text):
            """Thread-safe status update"""
            self.status_updated.emit(text)
        
        def update_result_safe(self, text):
            """Thread-safe result update"""
            self.result_updated.emit(text)
        
        def update_button_state_safe(self, start_enabled, stop_enabled):
            """Thread-safe button state update"""
            self.button_state_updated.emit(start_enabled, stop_enabled)
        
        def on_text_send(self):
            """Send text input"""
            text = self.text_input.text().strip()
            if text:
                self.text_input.clear()
                self.text_input_submitted.emit(text)
                self.update_status_safe("Status: Text input sent, processing...")
        
        def on_start(self):
            """Start recording"""
            self.is_recording = True
            self.update_button_state_safe(False, True)
            self.update_status_safe("Status: Recording...")
            self.result_text.clear()
        
        def on_stop(self):
            """Stop recording"""
            self.is_recording = False
            self.update_button_state_safe(True, False)
            self.update_status_safe("Status: Processing...")
else:
    class VoiceInputDialog:
        """Voice input dialog (placeholder class when PyQt5 unavailable)"""
        def __init__(self, parent=None):
            raise RuntimeError("PyQt5 not installed, cannot create GUI dialog")


if PYQT5_AVAILABLE and QObject is not None:
    class VoiceRecognizedSignal(QObject):
        """Voice recognition signal class"""
        recognized = pyqtSignal(str)
else:
    VoiceRecognizedSignal = None
