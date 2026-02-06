"""
Fish Speech voice synthesis module
Uses Fish Speech 1.5 for text-to-speech (local offline TTS)
Supports multiple languages

Deployment methods:
1. Direct use (requires cloning fish-speech GitHub repository)
2. API server mode (recommended, requires starting API server first)
"""
import sys
import os
from pathlib import Path
import threading
import queue
import time
import tempfile
import numpy as np
from typing import Optional

FISH_SPEECH_AVAILABLE = False
torch = None
tiktoken = None
requests = None

try:
    import torch
    FISH_SPEECH_AVAILABLE = True
except ImportError:
    FISH_SPEECH_AVAILABLE = False

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import requests
except ImportError:
    requests = None

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    snapshot_download = None
    hf_hub_download = None

PYGAME_AVAILABLE = False
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None


class FishSpeechVoiceSynthesis:
    """Fish Speech voice synthesis class
    
    Supports two deployment methods:
    1. API server mode (recommended): Call Fish Speech service via HTTP API
    2. Direct mode: Directly use Fish Speech code (requires cloning GitHub repository)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        language: str = "zh",
        device: Optional[str] = None,
        api_url: Optional[str] = None,
        use_api: bool = True
    ):
        """
        Initialize Fish Speech voice synthesis
        
        Args:
            model_path: Model file path (model.pth), only used in direct mode
            config_path: Config file path (config.json), only used in direct mode
            language: Language code (zh/en/ja, etc.)
            device: Device (cuda/cpu), only used in direct mode
            api_url: API server address (default: http://127.0.0.1:8028)
            use_api: Whether to use API mode (default True, recommended)
        """
        if not FISH_SPEECH_AVAILABLE:
            raise RuntimeError("PyTorch not installed, please install: pip install torch")
        
        self.use_api = use_api
        self.api_url = api_url or "http://127.0.0.1:8028"
        
        project_root = Path(__file__).parent.parent.parent
        self.models_dir = project_root / "models" / "fish-speech-1.5"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = model_path
        self.config_path = config_path
        self._language = language
        
        if use_api:
            if not self._check_api_available():
                raise RuntimeError(f"Fish Speech API server not available: {self.api_url}")
            self.model_available = True
            self.model = None
        else:
            if not HF_HUB_AVAILABLE:
                raise RuntimeError("huggingface_hub not installed, please install: pip install huggingface_hub")
            
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            self.model = None
            self.config = None
            self.tokenizer = None
            self._tts_engine = None
            self._tts_engine_lock = threading.Lock()
            
            try:
                self._load_model()
                self.model_available = True
            except Exception as e:
                self.model_available = False
                raise
        
        self.playback_queue = queue.Queue()
        self.is_playing = False
        self.running = True
        self.playback_thread = None
        
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
    
    def _check_api_available(self) -> bool:
        """Check if API server is available"""
        if not requests:
            return False
        
        health_url = f"{self.api_url}/v1/health"
        
        try:
            response = requests.get(health_url, timeout=3)
            return response.status_code == 200
        except Exception:
            return False
    
    def _load_model(self):
        """Load model and config"""
        if self.model_path is None:
            model_name = "fishaudio/fish-speech-1.5"
            try:
                model_path = snapshot_download(
                    repo_id=model_name,
                    local_dir=str(self.models_dir),
                    local_dir_use_symlinks=False
                )
                self.model_path = os.path.join(model_path, "model.pth")
                self.config_path = os.path.join(model_path, "config.json")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
    
    def set_language(self, language: str):
        """Set language"""
        self._language = language
    
    @property
    def language(self) -> str:
        """Get language"""
        return self._language
    
    def synthesize_and_play(self, text: str):
        """
        Synthesize voice and play (async)
        
        Args:
            text: Text to synthesize
        """
        if not text or len(text.strip()) == 0:
            return
        
        if not self.model_available:
            return
        
        self.playback_queue.put(("synthesize_and_play", text))
    
    def synthesize_and_play_streaming(self, text: str, is_final: bool = False):
        """
        Stream synthesis and play (async)
        
        Args:
            text: Current accumulated text
            is_final: Whether it's the final text
        """
        if is_final and text:
            self.synthesize_and_play(text)
    
    def reset_streaming_buffer(self):
        """Reset streaming buffer"""
        pass
    
    def _playback_loop(self):
        """Playback loop"""
        while self.running:
            try:
                task = self.playback_queue.get(timeout=0.5)
                if task is None:
                    break
                
                task_type, text = task
                
                if task_type == "synthesize_and_play":
                    try:
                        if self.use_api:
                            audio_data = self._synthesize_via_api(text)
                        else:
                            audio_data = self._synthesize_direct(text)
                        
                        if audio_data:
                            self.is_playing = True
                            self._play_audio(audio_data)
                            self.is_playing = False
                    except Exception:
                        pass
            except queue.Empty:
                continue
            except Exception:
                continue
    
    def _synthesize_via_api(self, text: str) -> Optional[bytes]:
        """Synthesize via API"""
        if not requests:
            return None
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/tts",
                json={
                    "text": text,
                    "language": self._language
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.content
        except Exception:
            pass
        
        return None
    
    def _synthesize_direct(self, text: str) -> Optional[bytes]:
        """Synthesize directly (requires Fish Speech code)"""
        return None
    
    def _play_audio(self, audio_data: bytes):
        """Play audio"""
        if not PYGAME_AVAILABLE:
            return
        
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(audio_data)
            temp_file.close()
            
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass
        except Exception:
            pass
    
    def start(self):
        """Start synthesis"""
        self.running = True
    
    def stop(self):
        """Stop synthesis"""
        self.running = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
