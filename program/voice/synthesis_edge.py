"""
Edge TTS voice synthesis module
Uses Microsoft Edge TTS for text-to-speech (online TTS)
Supports Chinese and English
"""
import threading
import queue
import time
import os
import tempfile
from pathlib import Path
from typing import Optional, Callable
import io

EDGE_TTS_AVAILABLE = False
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    edge_tts = None

PYGAME_AVAILABLE = False
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None

LANGUAGE_TO_VOICE = {
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "en-US": "en-US-AriaNeural",
    "en": "en-US-AriaNeural",
}


class EdgeVoiceSynthesis:
    """Edge TTS voice synthesis class"""
    
    def __init__(self, language: str = "zh-CN", voice: Optional[str] = None):
        """
        Initialize voice synthesis
        
        Args:
            language: Language code (zh-CN or en-US)
            voice: Voice name (optional, auto-selected based on language if not provided)
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts not installed, please install: pip install edge-tts")
        
        self.language = language
        self.voice = voice or LANGUAGE_TO_VOICE.get(language, "zh-CN-XiaoxiaoNeural")
        self.model_available = True
        
        self.playback_queue = queue.Queue()
        self.synthesized_queue = queue.Queue()
        self.is_playing = False
        self.running = True
        self.playback_thread = None
        
        self.sequence_counter = 0
        self.sequence_lock = threading.Lock()
        self.pending_audio = {}
        self.next_play_sequence = 0
        
        self.streaming_buffer = ""
        self.last_synthesized_length = 0
        self.streaming_lock = threading.Lock()
        
        self.synthesis_thread_pool = []
        self.max_synthesis_threads = 2
        self.synthesis_lock = threading.Lock()
        
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.prefetch_thread.start()
    
    def set_language(self, language: str):
        """Set language"""
        self.language = language
        self.voice = LANGUAGE_TO_VOICE.get(language, "zh-CN-XiaoxiaoNeural")
    
    def synthesize_and_play(self, text: str):
        """
        Synthesize voice and play (async, supports prefetch)
        
        Args:
            text: Text to synthesize
        """
        if not text or len(text.strip()) == 0:
            print("[Edge TTS] ⚠ Empty text, skipping synthesis")
            return
        
        print(f"[Edge TTS] Queuing synthesis request: '{text[:50]}...'")
        
        with self.sequence_lock:
            sequence = self.sequence_counter
            self.sequence_counter += 1
        
        self.playback_queue.put(("synthesize_and_play", text, sequence))
        
        if len(self.pending_audio) == 0 and not self.is_playing:
            self._start_synthesis_if_needed()
    
    def synthesize_and_play_streaming(self, text: str, is_final: bool = False):
        """
        Stream synthesis and play (async)
        Only triggers synthesis when encountering complete sentences
        
        Args:
            text: Current accumulated text (full text, not incremental)
            is_final: Whether it's the final text (streaming ended)
        """
        if not text or len(text.strip()) == 0:
            return
        
        with self.streaming_lock:
            new_text = text[self.last_synthesized_length:]
            
            if new_text:
                self.streaming_buffer += new_text
                self.last_synthesized_length = len(text)
            
            if is_final and self.streaming_buffer:
                buffer_to_synthesize = self.streaming_buffer
                self.streaming_buffer = ""
                self.last_synthesized_length = 0
                
                if buffer_to_synthesize.strip():
                    self.synthesize_and_play(buffer_to_synthesize)
            elif self.streaming_buffer:
                sentence_endings = ['。', '！', '？', '.', '!', '?', '\n']
                last_char = self.streaming_buffer[-1] if self.streaming_buffer else ''
                
                if last_char in sentence_endings:
                    sentences = []
                    current_sentence = ""
                    
                    for char in self.streaming_buffer:
                        current_sentence += char
                        if char in sentence_endings:
                            sentences.append(current_sentence.strip())
                            current_sentence = ""
                    
                    if sentences:
                        for sentence in sentences:
                            if sentence:
                                self.synthesize_and_play(sentence)
                        
                        if current_sentence:
                            self.streaming_buffer = current_sentence
                        else:
                            self.streaming_buffer = ""
    
    def reset_streaming_buffer(self):
        """Reset streaming buffer"""
        with self.streaming_lock:
            self.streaming_buffer = ""
            self.last_synthesized_length = 0
    
    def _start_synthesis_if_needed(self):
        """Start synthesis threads if needed (ensure not exceeding max thread count)"""
        with self.synthesis_lock:
            # Check current active synthesis thread count
            active_threads = sum(1 for t in self.synthesis_thread_pool if t.is_alive())
            
            # If there are idle threads and tasks in queue, start new synthesis threads
            while active_threads < self.max_synthesis_threads and not self.playback_queue.empty():
                try:
                    # Non-blocking get task
                    task = self.playback_queue.get_nowait()
                    if task:
                        if len(task) == 3:
                            task_type, text, sequence = task
                        else:
                            # Compatible with old format (no sequence)
                            task_type, text = task
                            with self.sequence_lock:
                                sequence = self.sequence_counter
                                self.sequence_counter += 1
                        
                        if task_type == "synthesize_and_play":
                            # Start synthesis thread, pass sequence
                            thread = threading.Thread(
                                target=self._synthesize_worker,
                                args=(text, sequence),
                                daemon=True
                            )
                            thread.start()
                            self.synthesis_thread_pool.append(thread)
                            active_threads += 1
                            print(f"[Edge TTS] Started synthesis thread, text: '{text[:50]}...', sequence: {sequence}, active threads: {active_threads}")
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"[Edge TTS] Failed to start synthesis thread: {e}")
                    break
            
            # Clean up completed threads
            self.synthesis_thread_pool = [t for t in self.synthesis_thread_pool if t.is_alive()]
    
    def _synthesize_worker(self, text: str, sequence: int):
        """Synthesis worker thread: synthesize audio and store in order"""
        try:
            print(f"[Edge TTS Synthesis Thread] Starting synthesis (sequence: {sequence}): '{text[:50]}...'")
            start_time = time.time()
            
            # Synthesize audio
            audio_data, file_path = self._synthesize_audio(text)
            
            elapsed_time = time.time() - start_time
            print(f"[Edge TTS Synthesis Thread] Synthesis completed (sequence: {sequence}), elapsed: {elapsed_time:.2f}s, audio size: {len(audio_data)} bytes")
            
            # Store in order to pending_audio dict (instead of directly putting in queue)
            with self.sequence_lock:
                self.pending_audio[sequence] = (audio_data, file_path, text, sequence)
                print(f"[Edge TTS Synthesis Thread] Audio stored (sequence: {sequence}), pending: {len(self.pending_audio)}, next play sequence: {self.next_play_sequence}")
            
        except Exception as e:
            print(f"[Edge TTS Synthesis Thread] Synthesis failed (sequence: {sequence}): {e}")
            import traceback
            traceback.print_exc()
    
    def _synthesize_audio(self, text: str):
        """Synthesize audio (without playing)"""
        import asyncio
        import tempfile
        
        # Use edge-tts to synthesize
        communicate = edge_tts.Communicate(text, self.voice)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_mp3_path = tmp_file.name
        
        # Save audio as MP3
        try:
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    communicate.save(tmp_mp3_path), loop
                )
                future.result(timeout=30)
            except RuntimeError:
                asyncio.run(communicate.save(tmp_mp3_path))
        except Exception as e:
            print(f"[Edge TTS] Async save failed: {e}")
            raise
        
        # Read audio data
        with open(tmp_mp3_path, "rb") as f:
            audio_data = f.read()
        
        return audio_data, tmp_mp3_path
    
    def _prefetch_loop(self):
        """Prefetch loop: prefetch next audio while playing"""
        print("[Edge TTS Prefetch Thread] Thread started")
        
        while self.running:
            try:
                # Check if prefetch is needed
                # If pending audio is less than 2 and text queue has tasks, start prefetch
                with self.sequence_lock:
                    pending_count = len(self.pending_audio)
                if pending_count < 2 and not self.playback_queue.empty():
                    self._start_synthesis_if_needed()
                
                time.sleep(0.1)  # Check every 0.1 seconds
            except Exception as e:
                print(f"[Edge TTS Prefetch Thread] Error: {e}")
                import traceback
                traceback.print_exc()
        
        print("[Edge TTS Prefetch Thread] Thread exited")
    
    def _playback_loop(self):
        """Playback loop: play synthesized audio in order"""
        print("[Edge TTS Playback Thread] Thread started")
        
        while self.running:
            try:
                # Check if there's next audio to play (in order)
                with self.sequence_lock:
                    if self.next_play_sequence in self.pending_audio:
                        # Found next audio to play
                        audio_data, file_path, text, sequence = self.pending_audio.pop(self.next_play_sequence)
                        self.next_play_sequence += 1
                        should_play = True
                    else:
                        should_play = False
                
                if should_play:
                    print(f"[Edge TTS Playback Thread] Playing audio (sequence: {sequence}), text: '{text[:50]}...', pending: {len(self.pending_audio)}")
                    
                    # Play audio
                    self.is_playing = True
                    self._play_audio(audio_data, file_path)
                    self.is_playing = False
                    
                    # After playback, trigger prefetch next (if there are tasks in text queue)
                    self._start_synthesis_if_needed()
                    
                    # Short delay so subtitle (output_loop) can update before next sentence plays; keeps subtitle in sync
                    time.sleep(0.03)
                    
                    # Delete temp file in background
                    threading.Thread(
                        target=self._delayed_delete,
                        args=(file_path, 0.5),
                        daemon=True
                    ).start()
                else:
                    # No audio to play, wait a bit
                    time.sleep(0.05)
                    # Check if there are tasks in text queue that need synthesis
                    if not self.playback_queue.empty():
                        self._start_synthesis_if_needed()
                    
            except Exception as e:
                print(f"[Edge TTS Playback Thread] Error: {e}")
                import traceback
                traceback.print_exc()
        
        print("[Edge TTS Playback Thread] Thread exited")
    
    def _delayed_delete(self, file_path: str, delay: float = 2.0):
        """Delete file after a delay (for Windows file locking issues)"""
        time.sleep(delay)
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            # Ignore errors in delayed deletion
            pass
    
    def _play_audio(self, audio_data: bytes, file_path: str):
        """Play audio"""
        print(f"[Edge TTS] Starting playback, file: {file_path}")
        
        if not PYGAME_AVAILABLE:
            print("[Edge TTS] ⚠ Pygame not available, trying pydub...")
            try:
                from pydub import AudioSegment
                from pydub.playback import play
                audio = AudioSegment.from_mp3(file_path)
                play(audio)
                print("[Edge TTS] ✓ Playback completed (using pydub)")
                return
            except ImportError:
                print("[Edge TTS] ✗ pydub not available, cannot play audio")
                return
            except Exception as e:
                print(f"[Edge TTS] ✗ pydub playback error: {e}")
                import traceback
                traceback.print_exc()
                return
        
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            print(f"[Edge TTS] Playing audio (pygame)...")
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.01)  # 0.01s poll for quicker transition to next sentence (match main_AIagent)
            
            print("[Edge TTS] ✓ Playback completed (using pygame)")
        except Exception as e:
            print(f"[Edge TTS] ✗ Playback error: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """Start synthesis"""
        self.running = True
    
    def stop(self):
        """Stop synthesis"""
        self.running = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
