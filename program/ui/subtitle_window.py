"""
Independent subtitle window
Simple implementation: directly displays text
Thread-safe: supports calling from background threads
"""
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont


class SubtitleWindow(QWidget):
    """Independent subtitle window"""
    
    _subtitle_signal = pyqtSignal(str, bool)
    
    def __init__(self):
        super().__init__()
        
        try:
            from PyQt5.QtCore import Qt as QtCore
            self._subtitle_signal.connect(self._set_subtitle_impl, QtCore.QueuedConnection)
            self._signal_connected = True
        except Exception:
            try:
                self._subtitle_signal.connect(self._set_subtitle_impl)
                self._signal_connected = True
            except Exception:
                self._signal_connected = False
        
        self.setWindowTitle("AI Desktop Pet - Subtitle")
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, False)
        
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        font = QFont("Consolas", 14, QFont.Normal)
        self.label.setFont(font)
        
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 200);
                padding: 10px;
                border-radius: 5px;
                border: 1px solid rgba(255, 255, 255, 50);
            }
        """)
        self.label.setVisible(True)
        
        font_metrics = self.label.fontMetrics()
        test_chinese_20 = "中文字符测试宽度计算中文字符测试宽度计算"
        width_chinese_20 = font_metrics.boundingRect(test_chinese_20).width()
        window_width = int(width_chinese_20 + 100)
        
        self.setFixedWidth(window_width)
        self.label.setMaximumWidth(window_width - 20)
        
        layout.addWidget(self.label)
        self.setLayout(layout)
        
        self.dragging = False
        self.drag_position = None
        
        self.setMinimumHeight(50)
        self.setMinimumWidth(window_width)
        self.resize(window_width, 50)
    
    def _set_initial_position(self):
        """Set initial window position (bottom right corner)"""
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        x = screen.width() - self.width() - 20
        y = screen.height() - self.height() - 20
        self.move(x, y)
        self._ensure_on_top()
    
    def _ensure_on_top(self):
        """Ensure window stays on top"""
        if not self.isVisible():
            self.setWindowFlags(
                Qt.WindowStaysOnTopHint |
                Qt.FramelessWindowHint |
                Qt.Tool
            )
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.show()
        
        self.raise_()
        
        if self.width() == 0 or self.height() == 0:
            self.resize(self.minimumWidth(), max(50, self.minimumHeight()))
    
    def mousePressEvent(self, event):
        """Mouse press event (for dragging)"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Mouse move event (for dragging)"""
        if self.dragging and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Mouse release event (end dragging)"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()
    
    def set_subtitle(self, text: str, append=False):
        """
        Set subtitle text
        Thread-safe: uses signal-slot mechanism
        
        Args:
            text: Subtitle text
            append: Whether to append (True) or replace (False, default)
        """
        if text is None or text == "":
            return
        
        try:
            from PyQt5.QtCore import QThread
            try:
                is_main_thread = QThread.currentThread() == QThread.mainThread()
            except:
                is_main_thread = False
            
            if is_main_thread:
                self._set_subtitle_impl(text, append)
            else:
                try:
                    if not hasattr(self, '_signal_connected') or not self._signal_connected:
                        try:
                            from PyQt5.QtCore import Qt as QtCore
                            try:
                                self._subtitle_signal.disconnect()
                            except:
                                pass
                            self._subtitle_signal.connect(self._set_subtitle_impl, QtCore.QueuedConnection)
                            self._signal_connected = True
                        except Exception:
                            pass
                    
                    self._subtitle_signal.emit(text, append)
                except Exception:
                    try:
                        from PyQt5.QtCore import QTimer
                        def callback():
                            try:
                                self._set_subtitle_impl(text, append)
                            except Exception:
                                pass
                        QTimer.singleShot(0, callback)
                    except Exception:
                        pass
        except Exception:
            try:
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._set_subtitle_impl(text, append))
            except Exception:
                pass
    
    def _set_subtitle_impl(self, text: str, append=False):
        """Internal method to set subtitle (called in main thread)"""
        try:
            if not hasattr(self, 'label'):
                return
            
            if text is None:
                return
            
            try:
                if not self.isVisible():
                    self._ensure_on_top()
                else:
                    self.raise_()
            except Exception:
                pass
            
            try:
                self.label.setVisible(True)
            except Exception:
                return
            
            if not append:
                try:
                    self.label.setText(text)
                except Exception:
                    return
            else:
                try:
                    current_text = self.label.text() or ""
                    if current_text == "":
                        self.label.setText(text)
                    else:
                        if text.startswith(current_text) or len(text) >= len(current_text):
                            self.label.setText(text)
                        else:
                            self.label.setText(text)
                except Exception:
                    return
            
            final_text = self.label.text()
            if final_text:
                self.label.setVisible(True)
            
            self.label.update()
            self._adjust_height()
            
            try:
                if not self.isVisible():
                    self._ensure_on_top()
                else:
                    self.raise_()
            except Exception:
                pass
            
            self.update()
            self.label.update()
        except Exception:
            pass
    
    def _adjust_height(self):
        """Adjust window height to fit text content"""
        try:
            font_metrics = self.label.fontMetrics()
            text_width = self.label.maximumWidth()
            text = self.label.text()
            
            if not text:
                self.setFixedHeight(50)
                return
            
            text_height = font_metrics.boundingRect(
                0, 0, text_width, 0,
                Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop,
                text
            ).height()
            
            padding = 30
            min_height = text_height + padding
            self.setFixedHeight(int(min_height))
        except Exception:
            pass
    
    def clear_subtitle(self):
        """Clear subtitle"""
        self.label.setText("")
        self.setFixedHeight(50)
