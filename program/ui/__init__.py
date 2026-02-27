"""
UI modules (subtitle window, voice input dialog, Live2D widget)
"""
from .subtitle_window import SubtitleWindow
from .voice_input_dialog import VoiceInputDialog, VoiceRecognizedSignal

__all__ = ['SubtitleWindow', 'VoiceInputDialog', 'VoiceRecognizedSignal']
