"""
Live2D renderer module
Uses Live2D models instead of static image rendering
"""
import os
import sys
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QTimerEvent, QRect
from PyQt5.QtGui import QPainter, QFont, QColor, QPen, QBrush, QFontMetrics

from utils.path_config import get_model_paths
from core.behavior import State

try:
    import live2d.v3 as live2d
    LIVE2D_AVAILABLE = True
except ImportError:
    LIVE2D_AVAILABLE = False
    live2d = None


class Live2DRenderer:
    """Live2D renderer"""
    
    def __init__(self, window=None):
        """
        Initialize Live2D renderer
        
        Args:
            window: Live2D window object (optional)
        """
        self.frame_count = 0
        self.window = window
        self.current_state = None
        
        self.model_path = None
        self._find_model_path()
        
        self.state_to_motion = {
            State.IDLE: ("Idle", 0),
            State.THINKING: ("Idle", 5),
            State.TALK: None,
        }
        
        self.talk_use_mixed = True
        
        self.state_to_expression = {
            State.IDLE: None,
            State.THINKING: None,
            State.TALK: None,
        }
        
        self.available_motion_groups = {}
        self.available_expressions = []
        
        self.state_descriptions = {
            State.IDLE: "Idle...",
            State.THINKING: "Thinking...",
            State.TALK: "Talking...",
        }
    
    def _find_model_path(self):
        """Find Live2D model path"""
        possible_paths = get_model_paths()
        
        for path in possible_paths:
            if os.path.exists(path):
                self.model_path = path
                return
    
    def set_window(self, window):
        """Set Live2D window"""
        self.window = window
    
    def render(self, state, force_restart=False):
        """
        Render current state
        
        Args:
            state: Current state (State enum)
            force_restart: Whether to force restart animation
        """
        self.frame_count += 1
        self.current_state = state
        
        if self.window:
            self.window.update_state(state, force_restart=force_restart)
    
    def clear(self):
        """Clear render content"""
        pass


class Live2DWidget(QOpenGLWidget):
    """Live2D OpenGL window"""
    
    def __init__(self, renderer):
        """
        Initialize Live2D window
        
        Args:
            renderer: Live2DRenderer instance
        """
        super().__init__()
        self.renderer = renderer
        self.current_state = None
        self.model = None
        self.live2d_initialized = False
        
        self.subtitle_text = ""
        self.subtitle_font = QFont("Arial", 14, QFont.Bold)
        self.subtitle_color = QColor(255, 255, 255)
        self.subtitle_bg_color = QColor(0, 0, 0, 180)
        self.subtitle_lines = []
        self.subtitle_scroll_offset = 0
        self.subtitle_max_lines = 10
        self.subtitle_line_spacing = 5
        self.subtitle_scroll_timer = 0
        self.subtitle_scroll_delay = 60
        self.subtitle_keep_timer = 0
        self.subtitle_keep_duration = 600
        
        self.setWindowTitle("AI Desktop Pet - Live2D")
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(400, 500)
        
        self.dragging = False
        self.drag_position = None
        
        self.startTimer(int(1000 / 60))
    
    def _set_initial_position(self):
        """Set initial window position (top right corner)"""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            screen = app.primaryScreen().geometry()
            x = screen.width() - self.width() - 20
            y = 20
            self.move(x, y)
    
    def initializeGL(self):
        """Initialize OpenGL context"""
        if not LIVE2D_AVAILABLE:
            return
        
        try:
            live2d.glInit()
        except Exception:
            return
        
        try:
            self.model = live2d.LAppModel()
        except Exception:
            return
        
        try:
            if self.renderer.model_path and os.path.exists(self.renderer.model_path):
                self.model.LoadModelJson(self.renderer.model_path)
                self.model.Resize(self.width(), self.height())
                
                try:
                    self._detect_model_resources()
                except Exception:
                    pass
            
            try:
                self.model.SetAutoBlinkEnable(True)
                self.model.SetAutoBreathEnable(True)
            except Exception:
                pass
            
            self.live2d_initialized = True
        except Exception:
            self.live2d_initialized = False
    
    def _detect_model_resources(self):
        """Detect available motion groups and expressions"""
        if not self.model:
            return
        
        try:
            motion_groups = self.model.GetMotionGroups()
            self.renderer.available_motion_groups = motion_groups
            
            expression_ids = self.model.GetExpressionIds()
            self.renderer.available_expressions = expression_ids
            
            self._update_state_mappings()
        except Exception:
            pass
    
    def _update_state_mappings(self):
        """Update state mappings based on model resources"""
        expressions = self.renderer.available_expressions
        if expressions:
            for exp_name in expressions:
                exp_lower = exp_name.lower()
                if "normal" in exp_lower or "smile" in exp_lower or "happy" in exp_lower or "f01" in exp_lower or "exp_01" in exp_lower:
                    if self.renderer.state_to_expression[State.IDLE] is None:
                        self.renderer.state_to_expression[State.IDLE] = exp_name
                elif "think" in exp_lower:
                    self.renderer.state_to_expression[State.THINKING] = exp_name
                elif "talk" in exp_lower or "speak" in exp_lower or "f02" in exp_lower or "exp_02" in exp_lower:
                    self.renderer.state_to_expression[State.TALK] = exp_name
    
    def resizeGL(self, w, h):
        """Called when window size changes"""
        if self.model:
            self.model.Resize(w, h)
    
    def timerEvent(self, event):
        """Timer event (triggers repaint)"""
        self.update()
    
    def paintGL(self):
        """Paint function (called every frame)"""
        if not self.model:
            return
        
        try:
            live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
            self.model.Update()
            self.model.Draw()
        except Exception:
            pass
        
        if self.subtitle_text:
            self._draw_subtitle()
    
    def _draw_subtitle(self):
        """Draw subtitle (after OpenGL rendering)"""
        if not self.subtitle_text and not self.subtitle_lines:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        painter.setFont(self.subtitle_font)
        font_metrics = painter.fontMetrics()
        
        if self.subtitle_text:
            self._update_subtitle_lines(painter)
        
        if not self.subtitle_lines:
            painter.end()
            return
        
        padding = 10
        max_width = self.width() - padding * 2 - 40
        line_height = font_metrics.height() + self.subtitle_line_spacing
        
        max_visible_lines = min(
            self.subtitle_max_lines,
            max(3, (self.height() - 100) // line_height)
        )
        
        display_lines = self.subtitle_lines[self.subtitle_scroll_offset:]
        if len(display_lines) > max_visible_lines:
            display_lines = display_lines[:max_visible_lines]
        
        if not display_lines:
            painter.end()
            return
        
        total_height = len(display_lines) * line_height - self.subtitle_line_spacing
        max_line_width = 0
        for line in display_lines:
            line_width = font_metrics.boundingRect(line).width()
            max_line_width = max(max_line_width, line_width)
        
        subtitle_x = (self.width() - max_line_width) // 2
        subtitle_y = self.height() - total_height - padding - 20
        
        bg_rect = QRect(
            subtitle_x - padding,
            subtitle_y - padding,
            max_line_width + padding * 2,
            total_height + padding * 2
        )
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.subtitle_bg_color))
        painter.drawRoundedRect(bg_rect, 5, 5)
        
        painter.setPen(QPen(self.subtitle_color))
        y_offset = subtitle_y
        for line in display_lines:
            line_x = (self.width() - font_metrics.boundingRect(line).width()) // 2
            painter.drawText(line_x, y_offset + font_metrics.ascent(), line)
            y_offset += line_height
        
        if len(self.subtitle_lines) > max_visible_lines:
            self.subtitle_scroll_timer += 1
            if self.subtitle_scroll_timer >= self.subtitle_scroll_delay:
                self.subtitle_scroll_timer = 0
                self.subtitle_scroll_offset += 1
                if self.subtitle_scroll_offset >= len(self.subtitle_lines) - max_visible_lines + 1:
                    self.subtitle_scroll_offset = 0
        
        painter.end()
        
        if self.subtitle_text or self.subtitle_lines:
            self.subtitle_keep_timer += 1
            if self.subtitle_keep_timer >= self.subtitle_keep_duration:
                self.clear_subtitle()
    
    def _update_subtitle_lines(self, painter=None):
        """Update subtitle line list (auto-wrap based on window width)"""
        if not self.subtitle_text:
            self.subtitle_lines = []
            return
        
        if painter is not None:
            font_metrics = painter.fontMetrics()
        else:
            from PyQt5.QtGui import QFontMetrics
            font_metrics = QFontMetrics(self.subtitle_font)
        
        padding = 10
        max_width = self.width() - padding * 2 - 40
        
        lines = []
        words = self.subtitle_text.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_width = font_metrics.boundingRect(test_line).width()
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                if font_metrics.boundingRect(word).width() > max_width:
                    lines.append(word)
                    current_line = ""
                else:
                    current_line = word
        
        if current_line:
            lines.append(current_line)
        
        if not lines:
            lines = [self.subtitle_text]
        
        self.subtitle_lines = lines
        self.subtitle_scroll_offset = 0
        self.subtitle_scroll_timer = 0
    
    def set_subtitle(self, text: str):
        """Set subtitle text"""
        self.subtitle_text = text
        self.subtitle_scroll_offset = 0
        self.subtitle_scroll_timer = 0
        self.subtitle_keep_timer = 0
        self.subtitle_lines = []
        self.update()
    
    def clear_subtitle(self, immediate=False):
        """Clear subtitle"""
        if immediate:
            self.subtitle_text = ""
            self.subtitle_lines = []
            self.subtitle_scroll_offset = 0
            self.subtitle_scroll_timer = 0
            self.subtitle_keep_timer = 0
            self.update()
    
    def update_state(self, state, force_restart=False):
        """Update displayed state"""
        old_window_state = self.current_state
        if state == self.current_state and not force_restart:
            return
        
        self.current_state = state
        
        if not self.model:
            return
        
        try:
            if state == State.TALK:
                import random
                if self.renderer.talk_use_mixed:
                    rand_val = random.randint(0, 9)
                    if rand_val <= 8:
                        self.model.StartMotion("Idle", rand_val, priority=3)
                    else:
                        self.model.StartMotion("TapBody", 0, priority=3)
                else:
                    self.model.StartRandomMotion("Idle", priority=3)
            else:
                motion_group, motion_no = self.renderer.state_to_motion.get(state, (None, None))
                
                if motion_group:
                    if motion_no is not None:
                        self.model.StartMotion(motion_group, motion_no, priority=3)
                    else:
                        self.model.StartRandomMotion(motion_group, priority=3)
            
            expression = self.renderer.state_to_expression.get(state)
            if expression:
                self.model.SetExpression(expression)
            elif state == State.IDLE:
                self.model.ResetExpression()
        except Exception:
            pass
    
    def mousePressEvent(self, event):
        """Mouse press event (for dragging and click detection)"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            
            if self.model:
                x = event.pos().x()
                y = event.pos().y()
                
                if self.model.HitTest("Body", x, y):
                    self.model.StartRandomMotion("TapBody", priority=4)
            
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Mouse move event (dragging window and gaze tracking)"""
        if self.dragging and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
        elif self.model:
            x = event.pos().x()
            y = event.pos().y()
            self.model.Drag(x, y)
    
    def mouseReleaseEvent(self, event):
        """Mouse release event"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()
    
    def closeEvent(self, event):
        """Window close event"""
        if self.live2d_initialized and LIVE2D_AVAILABLE:
            try:
                live2d.dispose()
            except:
                pass
        event.accept()
