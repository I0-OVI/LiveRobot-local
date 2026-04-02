"""
Voice modules (recognition and synthesis)
"""
from .recognition import VoiceRecognition, RecognitionResult, RecognitionStatus
from .synthesis_edge import EdgeVoiceSynthesis, EDGE_TTS_AVAILABLE
from .synthesis_fish import FishSpeechVoiceSynthesis, FISH_SPEECH_AVAILABLE

__all__ = [
    'VoiceRecognition', 'RecognitionResult', 'RecognitionStatus',
    'EdgeVoiceSynthesis', 'EDGE_TTS_AVAILABLE',
    'FishSpeechVoiceSynthesis', 'FISH_SPEECH_AVAILABLE'
]
