"""
Voice recognition module
Supports recording and speech-to-text functionality
"""
import speech_recognition as sr
import threading
import queue
import time
from typing import Optional, Callable
from enum import Enum
from dataclasses import dataclass
import os
import audioop


class RecognitionStatus(Enum):
    """Recognition status enumeration"""
    SUCCESS = "success"
    NO_SPEECH_DETECTED = "no_speech"
    RECOGNITION_FAILED = "recognition_failed"
    SERVICE_ERROR = "service_error"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class RecognitionResult:
    """Recognition result dataclass"""
    status: RecognitionStatus
    text: Optional[str] = None
    error_message: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Whether recognition was successful"""
        return self.status == RecognitionStatus.SUCCESS
    
    @property
    def is_no_speech(self) -> bool:
        """Whether no speech was detected"""
        return self.status == RecognitionStatus.NO_SPEECH_DETECTED
    
    @property
    def is_recognition_failed(self) -> bool:
        """Whether recognition failed (sound detected but cannot recognize)"""
        return self.status == RecognitionStatus.RECOGNITION_FAILED
    
    @property
    def is_service_error(self) -> bool:
        """Whether it's a service error"""
        return self.status == RecognitionStatus.SERVICE_ERROR


class VoiceRecognition:
    """Voice recognition class"""
    
    def __init__(self, 
                 language: str = "zh-CN",
                 energy_threshold: int = 100,
                 pause_threshold: float = 0.8,
                 phrase_threshold: float = 0.3):
        """
        Initialize voice recognizer
        
        Args:
            language: Recognition language, default Chinese (zh-CN), English uses "en-US"
            energy_threshold: Energy threshold for detecting speech start
            pause_threshold: Pause threshold (seconds) for detecting speech end
            phrase_threshold: Phrase threshold (seconds), minimum speech length
        """
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.language = language
        self.is_listening = False
        self.is_recording = False
        self.result_queue = queue.Queue()
        self.error_queue = queue.Queue()
        
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.phrase_threshold = phrase_threshold
        self.recognizer.dynamic_energy_threshold = True
        
        self._init_microphone()
    
    def _init_microphone(self):
        """Initialize microphone"""
        try:
            self.microphone = sr.Microphone()
        except Exception:
            self.microphone = None
    
    def adjust_for_ambient_noise(self, duration: float = 1.0):
        """
        Adjust for ambient noise
        
        Args:
            duration: Sampling duration (seconds), default 1.0
        """
        if not self.microphone:
            return
        
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        except Exception:
            pass
    
    def listen_once(self, timeout: Optional[float] = None, phrase_time_limit: Optional[float] = None, 
                   debug: bool = False) -> RecognitionResult:
        """
        Listen once for voice input and recognize
        
        Args:
            timeout: Timeout (seconds), None means no timeout
            phrase_time_limit: Maximum recording duration (seconds), None means unlimited
            debug: Whether to show debug info, default False
            
        Returns:
            RecognitionResult: Recognition result object
        """
        if not self.microphone:
            return RecognitionResult(
                status=RecognitionStatus.ERROR,
                error_message="Microphone not initialized"
            )
        
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                
                audio_analysis = self.analyze_audio_data(audio)
                
                if not audio_analysis['has_data']:
                    return RecognitionResult(
                        status=RecognitionStatus.ERROR,
                        error_message="Recorded audio data is empty"
                    )
                
                if audio_analysis['is_silence']:
                    return RecognitionResult(
                        status=RecognitionStatus.NO_SPEECH_DETECTED,
                        error_message="Recorded audio is silence or too quiet"
                    )
            
            try:
                text = self.recognizer.recognize_google(audio, language=self.language)
                return RecognitionResult(
                    status=RecognitionStatus.SUCCESS,
                    text=text
                )
            except sr.UnknownValueError:
                return RecognitionResult(
                    status=RecognitionStatus.RECOGNITION_FAILED,
                    error_message="Sound detected but cannot recognize content"
                )
            except sr.RequestError as e:
                return RecognitionResult(
                    status=RecognitionStatus.SERVICE_ERROR,
                    error_message=str(e)
                )
                
        except sr.WaitTimeoutError:
            return RecognitionResult(
                status=RecognitionStatus.NO_SPEECH_DETECTED,
                error_message="Listen timeout, no speech detected"
            )
        except Exception as e:
            return RecognitionResult(
                status=RecognitionStatus.ERROR,
                error_message=str(e)
            )
    
    def analyze_audio_data(self, audio) -> dict:
        """
        Analyze audio data
        
        Args:
            audio: speech_recognition AudioData object
            
        Returns:
            dict: Analysis result
        """
        result = {
            "has_data": False,
            "data_size": 0,
            "energy_level": 0.0,
            "is_silence": True,
            "sample_rate": 0,
            "duration": 0.0
        }
        
        try:
            audio_data = audio.get_raw_data()
            if not audio_data or len(audio_data) == 0:
                return result
            
            result["has_data"] = True
            result["data_size"] = len(audio_data)
            result["sample_rate"] = audio.sample_rate
            
            rms = audioop.rms(audio_data, 2)
            result["energy_level"] = float(rms)
            
            num_samples = len(audio_data) // 2
            result["duration"] = num_samples / audio.sample_rate
            
            result["is_silence"] = result["energy_level"] < 100
            
        except Exception:
            pass
        
        return result
