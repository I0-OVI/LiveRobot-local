"""
AI Agent Main Program
Integrates voice recognition, Qwen streaming inference, and state management

Workflow:
1. idle state -> voice recognition (Google API)
2. Use Qwen streaming inference (optional switch)
3. Detect language (Chinese/English)
4. Switch to talk state after voice ends
5. Output text
6. Return to idle state after output completes
7. Set state to thinking when text is still processing/voice recognition is in progress
"""
import sys
import os
import time
import threading
from enum import Enum
from typing import Optional, Callable
from queue import Queue

# Import path configuration utilities
from utils.path_config import get_current_dir, get_model_paths

# Import core behavior/state management
from core.behavior import BehaviorManager, State

# Import voice modules
from voice.recognition import VoiceRecognition, RecognitionResult, RecognitionStatus
from voice.synthesis_edge import EdgeVoiceSynthesis, EDGE_TTS_AVAILABLE
from voice.synthesis_fish import FishSpeechVoiceSynthesis, FISH_SPEECH_AVAILABLE

# Import AI text generation
from ai.text_generator import QwenTextGenerator

# Import UI modules
from ui import SubtitleWindow, VoiceInputDialog, VoiceRecognizedSignal

# Try to import and initialize Live2D (MUST be before QApplication creation)
LIVE2D_AVAILABLE = False
live2d_module = None

try:
    import live2d.v3 as live2d_module
    # Note: live2d.init() must be called before creating QApplication
    live2d_module.init()
    print("[Import] Live2D initialized successfully")
    LIVE2D_AVAILABLE = True
except ImportError:
    print("Warning: live2d.v3 module not installed")
    print("Please install: pip install live2d-py")
    LIVE2D_AVAILABLE = False
    live2d_module = None
except Exception as e:
    print(f"Warning: Live2D initialization failed: {e}")
    import traceback
    traceback.print_exc()
    LIVE2D_AVAILABLE = False
    live2d_module = None

# Try to import PyQt5
PYQT5_AVAILABLE = False
QApplication = None
QDialog = None
QVBoxLayout = None
QLabel = None
QPushButton = None
QTextEdit = None
Qt = None
QTimer = None
Live2DRenderer = None
Live2DWidget = None

try:
    from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QLineEdit
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QMetaObject, Q_ARG
    PYQT5_AVAILABLE = True
    print("[Import] PyQt5 imported successfully")
except ImportError as e:
    print(f"Warning: PyQt5 not installed, GUI features will be unavailable: {e}")
    PYQT5_AVAILABLE = False

# Import Live2D renderer (requires PyQt5 and live2d initialization)
if PYQT5_AVAILABLE and LIVE2D_AVAILABLE:
    try:
        from renderer.live2d_renderer import Live2DRenderer, Live2DWidget
        print("[Import] Live2D renderer imported successfully")
    except ImportError as e:
        print(f"Warning: Live2D renderer module import failed: {e}")
        import traceback
        traceback.print_exc()
        Live2DRenderer = None
        Live2DWidget = None
    except Exception as e:
        print(f"Warning: Live2D renderer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        Live2DRenderer = None
        Live2DWidget = None
else:
    if not PYQT5_AVAILABLE:
        print("[Import] Skipping Live2D renderer import (PyQt5 not available)")
    if not LIVE2D_AVAILABLE:
        print("[Import] Skipping Live2D renderer import (Live2D not initialized)")

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("Warning: langdetect not installed, language detection will be unavailable")
    print("Please install: pip install langdetect")
    LANGDETECT_AVAILABLE = False
    detect = None


class AgentState(Enum):
    """Internal agent state"""
    IDLE = "idle"              # Idle state, waiting for voice input
    LISTENING = "listening"    # Listening to voice
    THINKING = "thinking"      # Processing (voice recognition or text generation in progress)
    TALKING = "talking"        # Outputting text


class AIAgent:
    """AI Agent 主类"""
    
    def __init__(self, 
                 use_streaming: bool = True,
                 model_name: str = "Qwen/Qwen-7B-Chat",
                 model_cache_dir: Optional[str] = None,
                 language: str = "zh-CN",
                 use_gui: bool = True,
                 model_path: Optional[str] = None,
                 use_rag: bool = True,
                 # Replay configuration
                 replay_token_budget: int = 2000,
                 replay_persist_sessions: bool = True,
                 # RAG configuration
                 rag_top_k: int = 3,
                 rag_similarity_threshold: float = 0.7,
                 rag_summary_interval: int = 10,
                 rag_importance_base_threshold: float = 0.5,
                 rag_importance_max_memories: int = 1000,
                 rag_merge_similarity_threshold: float = 0.95,
                 rag_allow_no_memories: bool = True,
                 memory_persist_path: Optional[str] = None):
        """
        Initialize AI Agent
        
        Args:
            use_streaming: Whether to use streaming inference, default True
            model_name: Qwen model name
            model_cache_dir: Model cache directory
            language: Default recognition language (zh-CN or en-US)
            use_gui: Whether to use GUI interface, default True
            model_path: Live2D model path (optional)
            use_rag: Whether to enable memory system, default True
            replay_token_budget: Token budget for Replay system, default 2000
            replay_persist_sessions: Whether to persist Replay sessions, default True
            rag_top_k: Number of top memories to retrieve, default 3
            rag_similarity_threshold: Minimum similarity score for retrieval, default 0.7
            rag_summary_interval: Turns between summaries, default 10
            rag_importance_base_threshold: Base importance threshold, default 0.5
            rag_importance_max_memories: Max memories for adaptive threshold, default 1000
            rag_merge_similarity_threshold: Similarity threshold for merging, default 0.95
            rag_allow_no_memories: Allow no valid memories in RAG, default True
            memory_persist_path: Path to persist memory database, default "./memory_db"
        """
        self.use_streaming = use_streaming
        self.language = language
        self.use_gui = use_gui and PYQT5_AVAILABLE
        
        # State management
        self.behavior_manager = BehaviorManager()
        self.agent_state = AgentState.IDLE
        
        # Voice recognition
        self.voice_recognition = VoiceRecognition(language=language)
        
        # Voice synthesis (TTS)
        # Priority: 1. Fish Speech API mode 2. Edge TTS
        # Note: Local direct mode is temporarily commented out, pending debugging
        self.voice_synthesis = None
        
        # Convert language code
        lang_map = {
            "zh-CN": "zh",
            "zh": "zh",
            "en-US": "en",
            "en": "en",
            "ja": "ja",
            "jp": "ja",
        }
        fish_language = lang_map.get(language, "zh")
        
        # ===== Fish Speech 部分（已注释，暂时使用 Edge TTS） =====
        # # 1. 优先尝试使用 Fish Speech 本地直接模式（最低延迟）
        # if FISH_SPEECH_AVAILABLE:
        #     try:
        #         print("[初始化] 尝试初始化 Fish Speech 本地直接模式（优先，最低延迟）...")
        #         self.voice_synthesis = FishSpeechVoiceSynthesis(
        #             language=fish_language,
        #             use_api=False  # 使用本地直接模式
        #         )
        #         self.voice_synthesis.start()
        #         print("[初始化] ✓ Fish Speech 本地直接模式已初始化")
        #     except Exception as e:
        #         print(f"[初始化] ⚠ Fish Speech 本地直接模式初始化失败: {e}")
        #         print("[初始化] 将尝试 API 模式...")
        #         import traceback
        #         traceback.print_exc()
        #         self.voice_synthesis = None
        #
        # # 2. 如果本地直接模式失败，尝试 Fish Speech API 模式
        # if self.voice_synthesis is None and FISH_SPEECH_AVAILABLE:
        #     try:
        #         print("[初始化] 尝试初始化 Fish Speech API 模式（回退方案）...")
        #         self.voice_synthesis = FishSpeechVoiceSynthesis(
        #             language=fish_language,
        #             use_api=True  # 使用 API 模式
        #         )
        #         self.voice_synthesis.start()
        #         print("[初始化] ✓ Fish Speech API 模式已初始化")
        #     except Exception as e:
        #         print(f"[初始化] ⚠ Fish Speech API 模式初始化失败: {e}")
        #         print("[初始化] 将尝试 Edge TTS...")
        #         import traceback
        #         traceback.print_exc()
        #         self.voice_synthesis = None
        # ===== Fish Speech 部分结束 =====
        
        # 1. Use Edge TTS directly (Fish Speech is commented out)
        if EDGE_TTS_AVAILABLE:
            try:
                self.voice_synthesis = EdgeVoiceSynthesis(language=language)
                self.voice_synthesis.start()
                print("[Init] ✓ Edge TTS voice synthesis module initialized (fallback)")
            except Exception as e:
                print(f"[Init] ⚠ Edge TTS voice synthesis module initialization failed: {e}")
                import traceback
                traceback.print_exc()
                self.voice_synthesis = None
        
        if self.voice_synthesis is None:
            print("[Init] ⚠ Voice synthesis module unavailable (Fish Speech API mode and Edge TTS both failed)")
        
        # Qwen text generator
        if model_cache_dir is None:
            current_dir = get_current_dir()
            model_cache_dir = os.path.join(current_dir, "models", "qwen-7b-chat")
            # Ensure directory exists
            os.makedirs(model_cache_dir, exist_ok=True)
        
        print(f"[Init] Qwen model cache directory: {model_cache_dir}")
        print(f"[Init] Directory path: {os.path.abspath(model_cache_dir)}")
        print(f"[Info] Model will be permanently saved in this directory after download, won't be lost on restart")
        
        self.text_generator = QwenTextGenerator(
            model_name=model_name,
            cache_dir=model_cache_dir
        )
        
        # Live2D renderer (if available)
        self.renderer = None
        self.window = None
        self.subtitle_window = None  # Independent subtitle window
        self.app = None
        self.update_timer = None
        
        # Voice recognition signal (for inter-thread communication)
        self.voice_signal = None
        if self.use_gui and VoiceRecognizedSignal is not None:
            try:
                self.voice_signal = VoiceRecognizedSignal()
                self.voice_signal.recognized.connect(self._on_voice_recognized)
                print("[Init] ✓ Voice recognition signal created and connected")
            except Exception as e:
                print(f"[Init] ⚠ Voice recognition signal creation failed: {e}")
                self.voice_signal = None
        
        if self.use_gui and LIVE2D_AVAILABLE and Live2DRenderer is not None:
            try:
                print("[Init] Attempting to initialize Live2D renderer...")
                self.renderer = Live2DRenderer()
                
                # Set model path
                if model_path:
                    self.renderer.model_path = model_path
                    print(f"[Init] Using specified model path: {model_path}")
                else:
                    # Auto-detect model path
                    print("[Init] Auto-detecting Live2D model...")
                    self.renderer._find_model_path()
                    if self.renderer.model_path:
                        print(f"[Init] Found model path: {self.renderer.model_path}")
                    else:
                        print("[Warning] Live2D model file not found")
                        print("[Info] Please ensure model file exists, or specify via model_path parameter")
                
                print("[Init] Live2D renderer initialized successfully")
            except Exception as e:
                print(f"Warning: Live2D initialization failed: {e}")
                import traceback
                traceback.print_exc()
                self.renderer = None
                # Don't close GUI, just don't use Live2D
                print("Warning: Will continue running, but without Live2D rendering")
        
        # Voice input dialog
        self.voice_dialog = None
        
        # Conversation history (kept for backward compatibility)
        self.conversation_history = []
        
        # Memory Coordinator (Replay + RAG)
        self.use_rag = use_rag
        self.memory_coordinator = None
        self.replay_memory = None  # Fallback: Replay-only when RAG deps missing
        
        if self.use_rag:
            current_dir = get_current_dir()
            if memory_persist_path is None:
                memory_persist_path = os.path.join(current_dir, "memory_db")
            replay_persist_path = os.path.join(current_dir, "replay_db")
            
            try:
                from ai.memory import MemoryCoordinator
                
                # Initialize full memory coordinator (Replay + RAG)
                self.memory_coordinator = MemoryCoordinator(
                    replay_token_budget=replay_token_budget,
                    replay_persist_sessions=replay_persist_sessions,
                    replay_persist_path=replay_persist_path,
                    rag_persist_directory=memory_persist_path,
                    rag_top_k=rag_top_k,
                    rag_similarity_threshold=rag_similarity_threshold,
                    rag_summary_interval=rag_summary_interval,
                    rag_importance_base_threshold=rag_importance_base_threshold,
                    rag_importance_max_memories=rag_importance_max_memories,
                    rag_merge_similarity_threshold=rag_merge_similarity_threshold,
                    rag_allow_no_memories=rag_allow_no_memories,
                    rag_use_llm_trigger=False,
                    text_generator=None
                )
                
                print(f"[Init] ✓ Memory Coordinator initialized (Replay + RAG)")
                print(f"[Init] Replay path: {os.path.abspath(replay_persist_path)}, RAG path: {os.path.abspath(memory_persist_path)}")
                print(f"[Init] Replay token budget: {replay_token_budget}")
                print(f"[Init] RAG top_k: {rag_top_k}, similarity_threshold: {rag_similarity_threshold}")
                print(f"[Init] RAG summary interval: {rag_summary_interval}")
            except (ImportError, Exception) as e:
                print(f"[Init] ⚠ Full memory (RAG) not available: {e}")
                print("[Init] Replay-only mode: saving sessions to replay_db (install sentence-transformers for RAG)")
                # Fallback: Replay only (no sentence-transformers/chromadb needed)
                try:
                    from ai.memory.replay_memory import ReplayMemory  # Direct import to avoid loading RAG deps
                    self.replay_memory = ReplayMemory(
                        token_budget=replay_token_budget,
                        persist_sessions=replay_persist_sessions,
                        persist_path=replay_persist_path
                    )
                    print(f"[Init] ✓ Replay-only memory initialized at {os.path.abspath(replay_persist_path)}")
                except Exception as replay_e:
                    print(f"[Init] ⚠ Replay fallback failed: {replay_e}")
                    self.use_rag = False
        else:
            print("[Init] Memory system disabled")
        
        # Thread control
        self.running = False
        self.listening_thread = None
        self.stop_listening_event = threading.Event()
        self.manual_recording = False  # Manual recording mode
        
        # Text output queue
        self.text_output_queue = Queue()
        self.current_output_text = ""
        self.is_outputting = False
        self.output_thread = None  # Track output thread to avoid duplicate starts
        
        print("AI Agent initialization completed")
        print(f"Streaming inference: {'Enabled' if use_streaming else 'Disabled'}")
        print(f"Default language: {language}")
        print(f"GUI mode: {'Enabled' if self.use_gui else 'Disabled'}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect text language
        
        Args:
            text: Input text
        
        Returns:
            "zh-CN" or "en-US"
        """
        if not LANGDETECT_AVAILABLE or not text:
            return self.language  # Default to current language setting
        
        try:
            lang_code = detect(text)
            # langdetect returns ISO 639-1 codes (e.g., 'zh-cn', 'en')
            if lang_code.startswith('zh'):
                return "zh-CN"
            elif lang_code.startswith('en'):
                return "en-US"
            else:
                # Other languages default to current setting
                return self.language
        except LangDetectException:
            # Detection failed, use default language
            return self.language
    
    def _safe_set_state(self, agent_state: AgentState, behavior_state: str, force_immediate: bool = False):
        """
        Thread-safe state update method
        Ensures behavior_manager state updates are executed in the main thread
        
        Args:
            agent_state: AgentState enum value
            behavior_state: behavior_manager state string
            force_immediate: Whether to force immediate switch (don't wait for animation), default False
        """
        self.agent_state = agent_state
        if self.app and QTimer:
            # Update behavior_manager state in main thread
            if force_immediate:
                # Force immediate switch: clear queue and switch state immediately
                def do_force_switch():
                    self._force_immediate_state(behavior_state)
                QTimer.singleShot(0, do_force_switch)
                # Force process events to ensure callback executes immediately
                self.app.processEvents()
            else:
                QTimer.singleShot(0, lambda: self.behavior_manager.set_next_state(behavior_state))
        else:
            # If no GUI, update directly
            if force_immediate:
                self._force_immediate_state(behavior_state)
            else:
                self.behavior_manager.set_next_state(behavior_state)
    
    def _force_immediate_state(self, behavior_state: str):
        """
        Force immediate switch to specified state (don't wait for animation)
        
        Args:
            behavior_state: behavior_manager state string
        """
        # Clear state queue
        self.behavior_manager.clear_queue()
        
        # Convert string to State enum
        state_map = {
            "idle": State.IDLE,
            "talk": State.TALK,
            "thinking": State.THINKING,
        }
        state_str_lower = behavior_state.lower().strip()
        if state_str_lower not in state_map:
            print(f"Error: Invalid state '{behavior_state}'")
            return
        
        target_state = state_map[state_str_lower]
        old_state = self.behavior_manager.current_state
        
        # If target state is same as current, force restart animation
        if target_state == self.behavior_manager.current_state:
            self.behavior_manager._need_restart_animation = True
            # 重置状态持续时间，确保动画重新开始
            self.behavior_manager._reset_state_duration()
            return
        
        # 强制立即切换：直接调用 _transition_to_state
        # 先清除所有可能干扰的状态
        self.behavior_manager.target_state = None  # 清除目标状态，避免冲突
        self.behavior_manager.state_queue.clear()  # 确保队列为空
        
        # Directly call _transition_to_state to switch state
        # This sets current_state and calls _reset_state_duration
        self.behavior_manager._transition_to_state(target_state)
        
        # Verify state was actually switched
        if self.behavior_manager.current_state != target_state:
            # Force set state
            self.behavior_manager.current_state = target_state
            self.behavior_manager._reset_state_duration()
        
        # Mark animation restart needed
        self.behavior_manager._need_restart_animation = True
    
    def _on_voice_recognized(self, text: str):
        """
        Voice recognition success callback
        
        Args:
            text: Recognized text
        """
        print(f"\n{'='*60}")
        print(f"[Voice Recognition] ✓ Recognition successful")
        print(f"[Voice Recognition] Recognized text: '{text}'")
        print(f"[Voice Recognition] Text length: {len(text)} characters")
        
        # Check if text is empty
        if not text or len(text.strip()) == 0:
            print(f"[Voice Recognition] ⚠ Warning: Recognition result is empty, skipping generation")
            # Ensure state reset
            self.is_outputting = False
            self.current_output_text = ""
            self._safe_set_state(AgentState.IDLE, "idle")
            return
        
        # Ensure previous output is completed (avoid state confusion in multi-turn conversations)
        # Note: Don't wait here to avoid blocking main thread
        # If previous output not completed, force finish (avoid state stuck)
        if self.is_outputting:
            print(f"[Voice Recognition] ⚠ Warning: Previous output not completed, forcing finish")
            # Don't wait, force finish directly (avoid blocking GUI thread)
            self._finish_output()
        
        # Detect language
        detected_lang = self.detect_language(text)
        print(f"[Language Detection] Detected language: {detected_lang}")
        
        # If language changed, update recognizer and voice synthesis language
        if detected_lang != self.voice_recognition.language:
            self.voice_recognition.language = detected_lang
            print(f"[Language Switch] Switched to: {detected_lang}")
        
        # Synchronize voice synthesis language
        if self.voice_synthesis and detected_lang != self.voice_synthesis.language:
            self.voice_synthesis.set_language(detected_lang)
        
        # Voice recognition complete, stay in thinking state (already switched during recording)
        if self.agent_state != AgentState.THINKING:
            self._safe_set_state(AgentState.THINKING, "thinking")
        
        # Generate reply (will switch to talk state when output starts)
        print(f"[Process] Preparing to pass text to generation module...")
        print(f"[Process] Text to pass: '{text}'")
        print(f"[Process] Calling _generate_response...")
        try:
            self._generate_response(text)
            print(f"[Process] ✓ Text passed to generation module")
        except Exception as e:
            print(f"[Process] ✗ Error occurred while passing text: {e}")
            import traceback
            traceback.print_exc()
        print(f"{'='*60}\n")
    
    def _on_text_input_submitted(self, text: str):
        """
        Text input submission callback
        
        Args:
            text: User input text
        """
        print(f"\n{'='*60}")
        print(f"[Text Input] ✓ Text input received")
        print(f"[Text Input] Input text: '{text}'")
        print(f"[Text Input] Text length: {len(text)} characters")
        
        # Check if text is empty
        if not text or len(text.strip()) == 0:
            print(f"[Text Input] ⚠ Warning: Input text is empty, skipping generation")
            self._safe_set_state(AgentState.IDLE, "idle")
            return
        
        # Switch to thinking state
        self._safe_set_state(AgentState.THINKING, "thinking")
        
        # Update dialog status
        if self.voice_dialog:
            self.voice_dialog.update_status_safe("Status: Processing text input...")
        
        # Generate reply (will switch to talk state when output starts)
        print(f"[Process] Preparing to pass text to generation module...")
        print(f"[Process] Text to pass: '{text}'")
        print(f"[Process] Calling _generate_response...")
        try:
            self._generate_response(text)
            print(f"[Process] ✓ Text passed to generation module")
        except Exception as e:
            print(f"[Process] ✗ Error occurred while passing text: {e}")
            import traceback
            traceback.print_exc()
            # On error, return to idle state
            self._safe_set_state(AgentState.IDLE, "idle")
        print(f"{'='*60}\n")
    
    def _on_voice_error(self, result: RecognitionResult):
        """
        Voice recognition error callback
        
        Args:
            result: Recognition result object
        """
        print(f"\n[Voice Recognition] Error: {result.error_message}")
        
        # Recognition failed, return to idle state
        self._safe_set_state(AgentState.IDLE, "idle")
    
    def _generate_response(self, user_input: str):
        """
        Generate reply (using streaming or non-streaming)
        
        Args:
            user_input: User input text
        """
        # Check if input is empty
        if not user_input or len(user_input.strip()) == 0:
            self._safe_set_state(AgentState.IDLE, "idle")
            return
        
        # First detect keywords in user input, automatically trigger tool call
        print(f"[Keyword Detection] Detecting user input: '{user_input}'")
        tool_call = self.text_generator.tool_manager.detect_tool_call(user_input)
        
        if tool_call:
            tool_name, params = tool_call
            print(f"[Keyword Detection] ✓ Tool call detected: {tool_name}, params: {params}")
            print(f"[Keyword Detection] Executing tool directly, skipping AI generation")
            
            # Switch to thinking state
            self._safe_set_state(AgentState.THINKING, "thinking")
            
            # Execute tool
            tool_result = self.text_generator.tool_manager.execute_tool(tool_name, params)
            print(f"[Keyword Detection] Tool execution result: '{tool_result}'")
            
            # Switch to talk state and output result
            self._safe_set_state(AgentState.TALKING, "talk")
            
            # 输出工具结果
            if self.voice_synthesis:
                self.voice_synthesis.synthesize_and_play(tool_result)
            
            # 更新字幕
            if self.subtitle_window:
                self.subtitle_window.set_subtitle(tool_result)
            
            # 添加到对话历史
            self.conversation_history.append((user_input, tool_result))
            
            # Save to memory (Replay or Replay+RAG)
            if self.memory_coordinator:
                try:
                    clean_user_input = self.text_generator.tool_manager.remove_tool_markers(user_input)
                    clean_tool_result = self.text_generator.tool_manager.remove_tool_markers(tool_result)
                    self.memory_coordinator.save_conversation(
                        user_input=clean_user_input,
                        assistant_response=clean_tool_result,
                        importance=0.6,
                        use_llm_evaluation=False,
                        async_save=True
                    )
                    print(f"[Memory] Saved tool call to Replay (RAG in background)")
                except Exception as e:
                    print(f"[Memory] Error saving tool call: {e}")
            elif self.replay_memory:
                try:
                    clean_user_input = self.text_generator.tool_manager.remove_tool_markers(user_input)
                    clean_tool_result = self.text_generator.tool_manager.remove_tool_markers(tool_result)
                    self.replay_memory.add_turn(clean_user_input, clean_tool_result)
                    print(f"[Memory] Saved tool call to Replay (Replay-only mode)")
                except Exception as e:
                    print(f"[Memory] Error saving tool call: {e}")
            
            # Switch back to idle state
            self._safe_set_state(AgentState.IDLE, "idle")
            return
        
        # Ensure model is loaded
        if self.text_generator.model is None:
            try:
                self.text_generator.load_model()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._safe_set_state(AgentState.IDLE, "idle")
                return
        
        # Choose generation method
        if self.use_streaming:
            self._generate_response_streaming(user_input)
        else:
            self._generate_response_non_streaming(user_input)
    
    def _generate_response_streaming(self, user_input: str):
        """
        Generate reply using streaming inference (starts in thinking state, switches to talk and outputs after completion)
        
        Args:
            user_input: User input text
        """
        def stream_generation():
            try:
                # 确保在 thinking 状态下开始生成
                
                # 检查模型是否可用
                if self.text_generator.model is None:
                    self._safe_set_state(AgentState.IDLE, "idle")
                    raise RuntimeError("Model not loaded")
                
                full_response = ""
                first_chunk = True  # 标记是否是第一个片段
                
                # 重置语音合成的流式缓冲区
                if self.voice_synthesis:
                    self.voice_synthesis.reset_streaming_buffer()
                    # 检测语言并设置语音合成语言（使用第一个片段）
                    if first_chunk:
                        # 会在第一个片段时设置语言
                        pass
                
                # Get context from memory (Replay + RAG or Replay-only)
                replay_history = []
                enhanced_prompt = None
                if self.memory_coordinator:
                    try:
                        replay_history, rag_memories, enhanced_prompt, rag_used = self.memory_coordinator.get_context_for_generation(user_input)
                        self.conversation_history = replay_history
                        if enhanced_prompt:
                            system_prompt = self.text_generator._get_system_prompt(user_input)
                            enhanced_prompt = system_prompt + enhanced_prompt
                            print(f"[Memory] Using {len(replay_history)} replay turns and {len(rag_memories)} RAG memories")
                        else:
                            print(f"[Memory] Using {len(replay_history)} replay turns" + (", RAG triggered but no memories" if rag_used else ", RAG not triggered"))
                    except Exception as e:
                        print(f"[Memory] Error getting context: {e}")
                        import traceback
                        traceback.print_exc()
                        enhanced_prompt = None
                elif self.replay_memory:
                    replay_history = self.replay_memory.get_replay_history()
                    self.conversation_history = replay_history
                    print(f"[Memory] Using {len(replay_history)} replay turns (Replay-only mode)")
                
                # Start output thread early (before model inference) to avoid blocking when first token arrives
                self._start_text_output()
                
                try:
                    for response in self.text_generator.chat_stream(user_input, self.conversation_history, enhanced_prompt=enhanced_prompt):
                        full_response = response
                        
                        # 如果是第一个片段，切换到 talk 状态（output thread 已提前启动）
                        if first_chunk:
                            self._safe_set_state(AgentState.TALKING, "talk")
                            first_chunk = False
                            
                            # 设置语音合成语言
                            if self.voice_synthesis and response:
                                detected_lang = self.detect_language(response)
                                if detected_lang != self.voice_synthesis.language:
                                    self.voice_synthesis.set_language(detected_lang)
                        
                        # 将流式片段实时放入队列，用于字幕同步更新
                        if response and len(response.strip()) > 0:
                            # 移除工具标记后再输出
                            clean_response = self.text_generator.tool_manager.remove_tool_markers(response)
                            self.text_output_queue.put(("partial", clean_response))
                            
                            # 流式语音合成：在流式推理过程中就开始语音合成
                            # 每次增量文本到达时都触发语音合成检查
                            if self.voice_synthesis:
                                # 传递当前完整文本和是否结束标志（使用清理后的文本）
                                self.voice_synthesis.synthesize_and_play_streaming(clean_response, is_final=False)
                except AttributeError as e:
                    # 流式推理失败（通常是 transformers_stream_generator 兼容性问题）
                    if '_validate_model_class' in str(e):
                        # 回退到非流式推理
                        if self.memory_coordinator:
                            try:
                                replay_history, rag_memories, enhanced_prompt, rag_used = self.memory_coordinator.get_context_for_generation(user_input)
                                self.conversation_history = replay_history
                                if enhanced_prompt:
                                    system_prompt = self.text_generator._get_system_prompt(user_input)
                                    enhanced_prompt = system_prompt + enhanced_prompt
                            except Exception as e:
                                print(f"[Memory] Error getting context: {e}")
                                enhanced_prompt = None
                        elif self.replay_memory:
                            self.conversation_history = self.replay_memory.get_replay_history()
                        
                        response, updated_history = self.text_generator.chat(user_input, self.conversation_history, enhanced_prompt=enhanced_prompt)
                        self.conversation_history = updated_history
                        full_response = response
                        # 非流式回退时，切换状态（output thread 已提前启动）
                        if first_chunk:
                            self._safe_set_state(AgentState.TALKING, "talk")
                            first_chunk = False
                        # 将完整文本放入队列
                        self.text_output_queue.put(("complete", full_response))
                    else:
                        raise
                
                # 检查生成结果
                if not full_response or len(full_response.strip()) == 0:
                    raise RuntimeError("Generation result is empty")
                
                # 更新对话历史（保持向后兼容）
                if self.conversation_history:
                    self.conversation_history[-1] = (user_input, full_response)
                else:
                    self.conversation_history.append((user_input, full_response))
                
                # Save to memory (Replay or Replay+RAG)
                if self.memory_coordinator:
                    try:
                        clean_user_input = self.text_generator.tool_manager.remove_tool_markers(user_input)
                        clean_full_response = self.text_generator.tool_manager.remove_tool_markers(full_response)
                        self.memory_coordinator.save_conversation(
                            user_input=clean_user_input,
                            assistant_response=clean_full_response,
                            importance=None,
                            use_llm_evaluation=True,
                            async_save=True
                        )
                        print(f"[Memory] Saved conversation to Replay (RAG in background)")
                    except Exception as e:
                        print(f"[Memory] Error saving conversation: {e}")
                        import traceback
                        traceback.print_exc()
                elif self.replay_memory:
                    try:
                        clean_user_input = self.text_generator.tool_manager.remove_tool_markers(user_input)
                        clean_full_response = self.text_generator.tool_manager.remove_tool_markers(full_response)
                        self.replay_memory.add_turn(clean_user_input, clean_full_response)
                        print(f"[Memory] Saved conversation to Replay (Replay-only mode)")
                    except Exception as e:
                        print(f"[Memory] Error saving conversation: {e}")
                
                # 如果还没有切换到 talk 状态（非流式回退情况），现在切换
                # 注意：非流式回退时，文本已经在异常处理中放入队列了
                if first_chunk:
                    # 非流式回退情况：文本已在异常处理中放入队列
                    self._safe_set_state(AgentState.TALKING, "talk")
                    self._start_text_output()
                else:
                    # 正常流式完成：将完整结果放入队列（移除工具标记）
                    clean_full_response = self.text_generator.tool_manager.remove_tool_markers(full_response)
                    self.text_output_queue.put(("complete", clean_full_response))
                
                # 流式推理完成，发送最终文本给语音模块
                # 注意：在流式过程中已经分段合成了，这里只需要处理剩余部分
                if full_response and len(full_response.strip()) > 0 and self.voice_synthesis:
                    # 发送最终文本，标记为 is_final=True，确保所有剩余文本都被合成
                    clean_full_response = self.text_generator.tool_manager.remove_tool_markers(full_response)
                    self.voice_synthesis.synthesize_and_play_streaming(clean_full_response, is_final=True)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.text_output_queue.put(("error", str(e)))
                # 错误时回到 idle
                self._safe_set_state(AgentState.IDLE, "idle")
        
        # 在后台线程中生成
        generation_thread = threading.Thread(target=stream_generation, daemon=True)
        generation_thread.start()
    
    def _generate_response_non_streaming(self, user_input: str):
        """
        Generate reply using non-streaming inference (generates in thinking state, switches to talk after completion)
        
        Args:
            user_input: User input text
        """
        def generation():
            try:
                # 确保在 thinking 状态下生成
                
                # 检查模型是否可用
                if self.text_generator.model is None:
                    raise RuntimeError("Model not loaded")
                
                # Get context from memory (Replay + RAG or Replay-only)
                replay_history = []
                enhanced_prompt = None
                if self.memory_coordinator:
                    try:
                        replay_history, rag_memories, enhanced_prompt, rag_used = self.memory_coordinator.get_context_for_generation(user_input)
                        self.conversation_history = replay_history
                        if enhanced_prompt:
                            system_prompt = self.text_generator._get_system_prompt(user_input)
                            enhanced_prompt = system_prompt + enhanced_prompt
                            print(f"[Memory] Using {len(replay_history)} replay turns and {len(rag_memories)} RAG memories")
                        else:
                            print(f"[Memory] Using {len(replay_history)} replay turns" + (", RAG triggered but no memories" if rag_used else ", RAG not triggered"))
                    except Exception as e:
                        print(f"[Memory] Error getting context: {e}")
                        import traceback
                        traceback.print_exc()
                        enhanced_prompt = None
                elif self.replay_memory:
                    replay_history = self.replay_memory.get_replay_history()
                    self.conversation_history = replay_history
                    print(f"[Memory] Using {len(replay_history)} replay turns (Replay-only mode)")
                
                # Start output thread early (before model inference) to avoid blocking when result arrives
                self._start_text_output()
                
                response, updated_history = self.text_generator.chat(user_input, self.conversation_history, enhanced_prompt=enhanced_prompt)
                
                self.conversation_history = updated_history
                
                # 检查生成结果
                if not response or len(response.strip()) == 0:
                    raise RuntimeError("Generation result is empty")
                
                # Save to memory (Replay or Replay+RAG)
                if self.memory_coordinator:
                    try:
                        clean_user_input = self.text_generator.tool_manager.remove_tool_markers(user_input)
                        clean_response = self.text_generator.tool_manager.remove_tool_markers(response)
                        self.memory_coordinator.save_conversation(
                            user_input=clean_user_input,
                            assistant_response=clean_response,
                            importance=None,
                            use_llm_evaluation=True,
                            async_save=True
                        )
                        print(f"[Memory] Saved conversation to Replay (RAG in background)")
                    except Exception as e:
                        print(f"[Memory] Error saving conversation: {e}")
                        import traceback
                        traceback.print_exc()
                elif self.replay_memory:
                    try:
                        clean_user_input = self.text_generator.tool_manager.remove_tool_markers(user_input)
                        clean_response = self.text_generator.tool_manager.remove_tool_markers(response)
                        self.replay_memory.add_turn(clean_user_input, clean_response)
                        print(f"[Memory] Saved conversation to Replay (Replay-only mode)")
                    except Exception as e:
                        print(f"[Memory] Error saving conversation: {e}")
                
                # 生成完成后，切换到 talk 状态（output thread 已提前启动）
                self._safe_set_state(AgentState.TALKING, "talk")
                
                # 移除工具标记后再输出
                clean_response = self.text_generator.tool_manager.remove_tool_markers(response)
                
                # 将完整文本放入队列
                self.text_output_queue.put(("complete", clean_response))
                
                # 文字生成完成后，立即发送给语音模块进行语音合成
                # 这样可以在文字显示的同时开始语音合成，减少延迟
                if clean_response and len(clean_response.strip()) > 0:
                    if self.voice_synthesis:
                        # 检测语言并设置语音合成语言（使用清理后的文本）
                        detected_lang = self.detect_language(clean_response)
                        if detected_lang != self.voice_synthesis.language:
                            self.voice_synthesis.set_language(detected_lang)
                        
                        # 异步合成并播放语音（不阻塞文本输出，使用清理后的文本）
                        print(f"[Voice Output] Text generation completed, starting voice synthesis: '{clean_response[:50]}...'")
                        self.voice_synthesis.synthesize_and_play(clean_response)
                    else:
                        print(f"[Voice Output] ⚠ Voice synthesis module unavailable, skipping voice output (piper-tts not installed or initialization failed)")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.text_output_queue.put(("error", str(e)))
                # 错误时回到 idle
                self._safe_set_state(AgentState.IDLE, "idle")
        
        # 在后台线程中生成
        generation_thread = threading.Thread(target=generation, daemon=True)
        generation_thread.start()
    
    def _start_text_output(self):
        """
        Start text output
        """
        # If output is in progress, force finish previous output first
        if self.is_outputting:
            print(f"[Output] ⚠ Warning: Output in progress, forcing finish of previous output")
            # Check if thread is still running
            if self.output_thread and self.output_thread.is_alive():
                print(f"[Output] ⚠ Output thread still running: {self.output_thread.name} (ID: {self.output_thread.ident})")
            # 强制完成之前的输出
            self._finish_output()
            # 等待一小段时间让线程退出（不阻塞太久）
            if self.output_thread and self.output_thread.is_alive():
                for i in range(20):  # 最多等待2秒（20 * 0.1秒），增加等待时间
                    if not self.output_thread.is_alive():
                        print(f"[Output] ✓ Old thread exited (waited {i * 0.1:.1f} seconds)")
                        break
                    time.sleep(0.1)
                if self.output_thread.is_alive():
                    print(f"[Output] ⚠ Warning: Output thread did not exit within 2 seconds, continuing: {self.output_thread.name} (ID: {self.output_thread.ident})")
        
        # Ensure previous output thread has ended (double check)
        if self.output_thread and self.output_thread.is_alive():
            print(f"[Output] ⚠ Warning: Found residual output thread, forcing reset state: {self.output_thread.name} (ID: {self.output_thread.ident})")
            # 强制重置状态，让旧线程退出
            old_thread = self.output_thread
            self.is_outputting = False
            # 等待更长时间，确保线程退出
            for i in range(10):  # Wait 1 second
                if not old_thread.is_alive():
                    print(f"[Output] ✓ Residual thread exited (waited {i * 0.1:.1f} seconds)")
                    break
                time.sleep(0.1)
            if old_thread.is_alive():
                print(f"[Output] ⚠ Warning: Residual thread still not exited: {old_thread.name} (ID: {old_thread.ident})")
            # 将旧线程引用设为 None，避免干扰
            self.output_thread = None
        
        # 彻底清空队列，避免累积未处理的消息（多轮对话后可能残留）
        cleared_count = 0
        max_clear_attempts = 100  # 最多尝试清空100次，避免无限循环
        clear_attempts = 0
        while not self.text_output_queue.empty() and clear_attempts < max_clear_attempts:
            try:
                self.text_output_queue.get_nowait()
                cleared_count += 1
                clear_attempts += 1
            except:
                break
        if cleared_count > 0:
            print(f"[Output] Cleared queue (cleared {cleared_count} messages), ready to start new output")
        else:
            print(f"[Output] Queue is empty, ready to start new output")
        
        # 重置状态
        self.is_outputting = True
        self.current_output_text = ""  # 重置当前输出文本
        
        # 注意：不再清除字幕，而是直接替换为新内容
        # 这样可以避免清除字幕时的崩溃问题
        
        def output_loop():
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name
            print(f"[Output Thread] Thread started: {thread_name} (ID: {thread_id})")
            output_timeout = 0  # 输出超时计数器
            max_timeout = 300  # 最大等待时间（30秒，0.1秒 * 300）
            
            try:
                while self.is_outputting:
                    try:
                        # 从队列获取文本（短超时减少首字延迟）
                        try:
                            msg_type, text = self.text_output_queue.get(timeout=0.03)
                            output_timeout = 0  # 重置超时计数器
                            
                            if msg_type == "partial":
                                # 流式输出增量文本（追加模式，不清除旧字幕）
                                self.current_output_text = text
                                if self.subtitle_window:
                                    # 使用信号槽机制，自动在主线程中执行，无需手动处理事件
                                    self.subtitle_window.set_subtitle(text, append=True)
                                print(f"\r[Output] {text}", end="", flush=True)
                            
                            elif msg_type == "complete":
                                # 输出完成
                                self.current_output_text = text
                                # 确保字幕显示最终完整文本（使用替换模式，不清除）
                                if self.subtitle_window:
                                    # 使用信号槽机制，自动在主线程中执行，无需手动处理事件
                                    self.subtitle_window.set_subtitle(text, append=False)  # 使用替换模式
                                print(f"{text}")
                                
                                # 注意：语音合成已在文字生成完成后立即触发（在_generate_response_streaming或_generate_response_non_streaming中）
                                # 这里不再重复触发，避免重复合成
                                
                                print(f"[Output Thread] Received complete message, preparing to exit: {thread_name} (ID: {thread_id})")
                                self._finish_output()
                                return  # 确保退出循环
                            
                            elif msg_type == "error":
                                print(f"[Output Thread] Received error message, preparing to exit: {thread_name} (ID: {thread_id})")
                                self._finish_output()
                                return  # 确保退出循环
                            
                        except:
                            # 队列为空，继续等待
                            output_timeout += 1
                            if output_timeout >= max_timeout:
                                # Timeout, force finish output
                                print(f"[Output Thread] Timeout, forcing exit: {thread_name} (ID: {thread_id})")
                                self._finish_output()
                                return
                            time.sleep(0.02)  # 20ms poll for lower latency
                            
                    except Exception as e:
                        print(f"[Output Thread] Exception, preparing to exit: {thread_name} (ID: {thread_id}), error: {e}")
                        import traceback
                        traceback.print_exc()
                        self._finish_output()
                        return  # 确保退出循环
            finally:
                print(f"[Output Thread] Thread exited: {thread_name} (ID: {thread_id})")
        
        # 创建并启动输出线程（使用唯一名称避免冲突）
        thread_name = f"output_loop_{threading.current_thread().ident}_{time.time()}"
        self.output_thread = threading.Thread(target=output_loop, daemon=True, name=thread_name)
        self.output_thread.start()
        print(f"[Output] ✓ Output thread started: {self.output_thread.name} (ID: {self.output_thread.ident})")
    
    def _finish_output(self):
        """
        Finish text output and return to idle state
        Workflow: talk -> idle
        """
        if not self.is_outputting:
            # 如果已经完成，避免重复调用
            print(f"[完成输出] ⚠ 输出已完成，跳过重复调用")
            return
        
        
        # 清空队列，确保没有残留消息（多轮对话后可能累积）
        queue_size = 0
        while not self.text_output_queue.empty():
            try:
                self.text_output_queue.get_nowait()
                queue_size += 1
            except:
                break
        if queue_size > 0:
            print(f"[完成输出] 清空了 {queue_size} 个队列中的残留消息")
        
        # 重置状态
        self.is_outputting = False
        self.current_output_text = ""
        
        # 确保切换到 idle 状态（强制立即切换，不等待动画完成）
        # 工作流：talk -> idle
        
        # 先设置 agent_state 为 IDLE
        self.agent_state = AgentState.IDLE
        
        # 直接在主线程中强制切换状态（不通过QTimer，避免延迟）
        if self.app and QTimer:
            # 使用 QTimer.singleShot 确保在主线程中执行
            def do_force_switch():
                # Ensure behavior_manager also switches to idle
                self._force_immediate_state("idle")
                # Verify state
                if self.behavior_manager.current_state != State.IDLE:
                    self.behavior_manager.current_state = State.IDLE
                    self.behavior_manager._reset_state_duration()
                    self.behavior_manager._need_restart_animation = True
            QTimer.singleShot(0, do_force_switch)
            # 强制处理事件，确保回调立即执行
            self.app.processEvents()
        else:
            # If no GUI, call directly
            self._force_immediate_state("idle")
            # Verify state
            if self.behavior_manager.current_state != State.IDLE:
                self.behavior_manager.current_state = State.IDLE
                self.behavior_manager._reset_state_duration()
                self.behavior_manager._need_restart_animation = True
        
    
    def _verify_idle_state(self):
        """Verify idle state is correctly set"""
        if self.agent_state != AgentState.IDLE:
            self.agent_state = AgentState.IDLE
            if self.behavior_manager:
                self.behavior_manager.set_next_state("idle")
                if self.app:
                    self.app.processEvents()
    
    def _listening_loop(self):
        """
        Voice listening loop
        """
        self.agent_state = AgentState.LISTENING
        
        while self.running and not self.stop_listening_event.is_set():
            try:
                # 如果当前不在 IDLE 或 LISTENING 状态，等待状态恢复
                if self.agent_state not in [AgentState.IDLE, AgentState.LISTENING]:
                    while self.agent_state != AgentState.IDLE and self.running:
                        time.sleep(0.1)
                    if not self.running:
                        break
                
                # 设置为 listening 状态
                self.agent_state = AgentState.LISTENING
                
                # 监听一次语音
                result = self.voice_recognition.listen_once(
                    timeout=5.0,  # 5秒超时
                    phrase_time_limit=10.0  # 最长10秒语音
                )
                
                if result.is_success:
                    # Recognition successful, switch to thinking state
                    self._safe_set_state(AgentState.THINKING, "thinking")
                    
                    # 如果启用流式推理，在 thinking 状态下开始生成
                    if self.use_streaming:
                        # 在后台线程中开始流式生成
                        print(f"[Listening Loop] Using streaming inference, preparing to generate")
                        print(f"[Listening Loop] Recognized text: '{result.text}'")
                        def start_streaming():
                            print(f"[Listening Loop] Streaming generation thread started")
                            self._generate_response_streaming(result.text)
                        threading.Thread(target=start_streaming, daemon=True).start()
                    else:
                        # Non-streaming: generate directly (will switch to talk when output starts)
                        print(f"[Listening Loop] Using non-streaming inference, preparing to generate")
                        print(f"[Listening Loop] Recognized text: '{result.text}'")
                        self._on_voice_recognized(result.text)
                
                elif result.is_no_speech:
                    # No speech detected, continue listening (stay in listening state)
                    print("[Listening] No speech detected, continuing to listen...")
                    continue
                else:
                    # 识别失败，回到 idle
                    self._on_voice_error(result)
                
                # Wait for state to return to IDLE before continuing
                while self.agent_state != AgentState.IDLE and self.running:
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"\n[Listening Error] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
        
        print("\n[Listening] Listening loop ended")
    
    def _init_gui(self):
        """Initialize GUI"""
        if not self.use_gui:
            return
        
        try:
            # 确保只有一个 QApplication 实例
            if not QApplication.instance():
                self.app = QApplication(sys.argv)
            else:
                self.app = QApplication.instance()
            
            # 创建 Live2D 窗口
            if self.renderer and LIVE2D_AVAILABLE and Live2DWidget is not None:
                try:
                    # 检查模型路径是否存在
                    if not self.renderer.model_path:
                        print("[GUI] Warning: Live2D model path not set, attempting to re-detect...")
                        self.renderer._find_model_path()
                    
                    if self.renderer.model_path:
                        model_path_abs = os.path.abspath(self.renderer.model_path)
                        print(f"[GUI] Checking model path: {model_path_abs}")
                        print(f"[GUI] Path exists: {os.path.exists(model_path_abs)}")
                        
                        if os.path.exists(model_path_abs):
                            print(f"[GUI] Creating Live2D window, model path: {model_path_abs}")
                            
                            # Create step by step, catch exceptions at each step
                            try:
                                print("[GUI] Step 1: Creating Live2DWidget object...")
                                self.window = Live2DWidget(self.renderer)
                                print("[GUI] ✓ Live2DWidget created successfully")
                            except Exception as e:
                                print(f"[GUI] ✗ Live2DWidget creation failed: {e}")
                                import traceback
                                traceback.print_exc()
                                self.window = None
                                raise
                            
                            try:
                                print("[GUI] Step 2: Setting window properties...")
                                self.renderer.set_window(self.window)
                                self.window.setWindowTitle("Live2D AI Agent")
                                self.window.resize(800, 600)
                                print("[GUI] ✓ Window properties set successfully")
                            except Exception as e:
                                print(f"[GUI] ✗ Window property setting failed: {e}")
                                import traceback
                                traceback.print_exc()
                                raise
                            
                            try:
                                print("[GUI] Step 3: Showing window...")
                                self.window.show()
                                print("[GUI] ✓ Window show call successful")
                            except Exception as e:
                                print(f"[GUI] ✗ Window show failed: {e}")
                                import traceback
                                traceback.print_exc()
                                raise
                            
                            try:
                                print("[GUI] Step 4: Setting window position...")
                                self.window._set_initial_position()
                                print("[GUI] ✓ Window position set successfully")
                            except Exception as e:
                                print(f"[GUI] ⚠ Window position setting warning: {e} (continuing)")
                                import traceback
                                traceback.print_exc()
                            
                            try:
                                print("[GUI] Step 5: Processing events...")
                                self.window.raise_()
                                self.app.processEvents()
                                print("[GUI] ✓ Event processing successful")
                            except Exception as e:
                                print(f"[GUI] ✗ Event processing failed: {e}")
                                import traceback
                                traceback.print_exc()
                                raise
                            
                            # Wait for OpenGL initialization (initializeGL will be called automatically after show())
                            import time
                            print("[GUI] Step 6: Waiting for OpenGL initialization...")
                            for i in range(10):  # Wait up to 1 second
                                self.app.processEvents()
                                time.sleep(0.1)
                                if hasattr(self.window, 'live2d_initialized') and self.window.live2d_initialized:
                                    print("[GUI] ✓ OpenGL initialization completed")
                                    break
                            
                            print("[GUI] Live2D window creation completed")
                            print(f"[GUI] Window visibility: {self.window.isVisible()}")
                            print(f"[GUI] Window size: {self.window.width()}x{self.window.height()}")
                            print(f"[GUI] Window position: ({self.window.x()}, {self.window.y()})")
                            
                            # Check if Live2D is initialized
                            if hasattr(self.window, 'live2d_initialized'):
                                print(f"[GUI] Live2D initialization status: {self.window.live2d_initialized}")
                            else:
                                print("[GUI] Warning: live2d_initialized attribute does not exist")
                            
                        else:
                            print(f"[GUI] Error: Live2D model file does not exist")
                            print(f"[GUI] Model path: {model_path_abs}")
                            print("[GUI] Please check if model file exists, or specify correct path via model_path parameter")
                            self.window = None
                    else:
                        print("[GUI] Error: Live2D model path not found")
                        print("[GUI] Please ensure model file exists, or specify via model_path parameter")
                        self.window = None
                except Exception as e:
                    print(f"[GUI] ✗ Live2D window creation process error: {e}")
                    import traceback
                    traceback.print_exc()
                    self.window = None
                    # Keep renderer, allow retry later
                    print("[GUI] Warning: Will continue running, but without Live2D window")
            
            # Create independent subtitle window
            try:
                print("[GUI] Creating subtitle window...")
                self.subtitle_window = SubtitleWindow()
                
                # Set position first, then show
                self.subtitle_window._set_initial_position()
                
                # Show window
                self.subtitle_window.show()
                self.subtitle_window.raise_()  # Ensure window appears on top
                self.subtitle_window.activateWindow()  # Activate window
                
                # Process events to ensure window is actually shown
                self.app.processEvents()
                
                # Ensure window is visible again
                if not self.subtitle_window.isVisible():
                    print("[GUI] ⚠ Subtitle window not visible after creation, attempting to re-show...")
                    self.subtitle_window.show()
                    self.subtitle_window.raise_()
                    self.app.processEvents()
                
                # Clear test subtitle (if any), wait for actual subtitle update
                # No longer set test subtitle, let actual subtitle display naturally
                self.app.processEvents()
                
                print(f"[GUI] ✓ Subtitle window created successfully")
                print(f"[GUI]   Visibility: {self.subtitle_window.isVisible()}")
                print(f"[GUI]   Position: ({self.subtitle_window.x()}, {self.subtitle_window.y()})")
                print(f"[GUI]   Size: {self.subtitle_window.width()}x{self.subtitle_window.height()}")
                print(f"[GUI]   Label text: '{self.subtitle_window.label.text()}'")
                print(f"[GUI]   Label visible: {self.subtitle_window.label.isVisible()}")
            except Exception as e:
                print(f"[GUI] ✗ Subtitle window creation failed: {e}")
                import traceback
                traceback.print_exc()
                self.subtitle_window = None
            
            # Create voice and text input dialog
            if VoiceInputDialog is not None:
                try:
                    self.voice_dialog = VoiceInputDialog()
                    # Connect dialog signals
                    self.voice_dialog.start_button.clicked.connect(self._on_dialog_start_recording)
                    self.voice_dialog.stop_button.clicked.connect(self._on_dialog_stop_recording)
                    # Connect text input signal
                    self.voice_dialog.text_input_submitted.connect(self._on_text_input_submitted)
                    print("Input dialog created successfully (supports text and voice input)")
                except Exception as e:
                    print(f"Warning: Input dialog creation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self.voice_dialog = None
            
            # Create update timer
            if self.renderer and QTimer is not None:
                try:
                    self.update_timer = QTimer()
                    self.update_timer.timeout.connect(self._update_loop)
                    self.update_timer.start(100)  # Update every 100ms
                    print("Update timer created successfully")
                except Exception as e:
                    print(f"Warning: Update timer creation failed: {e}")
                    self.update_timer = None
            
            print("GUI initialization completed")
        except Exception as e:
            print(f"GUI initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.use_gui = False
            self.app = None
            self.window = None
            self.voice_dialog = None
            self.update_timer = None
    
    def _update_loop(self):
        """Update loop (used in GUI mode)"""
        if not self.running:
            return
        
        try:
            # Update behavior state
            self.behavior_manager.update(0.1)
            
            
            # 渲染当前状态
            if self.renderer:
                current_state = self.behavior_manager.get_current_state()
                # Enhanced state synchronization: ensure agent_state and behavior_manager.current_state are consistent
                # Workflow: idle -> thinking -> talk -> idle
                state_synced = False
                if self.agent_state == AgentState.IDLE:
                    if current_state != State.IDLE:
                        self._force_immediate_state("idle")
                        current_state = self.behavior_manager.get_current_state()
                        state_synced = True
                elif self.agent_state == AgentState.TALKING:
                    if current_state != State.TALK:
                        self._force_immediate_state("talk")
                        current_state = self.behavior_manager.get_current_state()
                        state_synced = True
                elif self.agent_state == AgentState.THINKING:
                    if current_state != State.THINKING:
                        self._force_immediate_state("thinking")
                        current_state = self.behavior_manager.get_current_state()
                        state_synced = True
                
                # If state was synced, verify sync result (may still be inconsistent after multiple rounds)
                if state_synced:
                    final_state = self.behavior_manager.get_current_state()
                    if (self.agent_state == AgentState.IDLE and final_state != State.IDLE) or \
                       (self.agent_state == AgentState.TALKING and final_state != State.TALK) or \
                       (self.agent_state == AgentState.THINKING and final_state != State.THINKING):
                        # Force reset to correct state
                        if self.agent_state == AgentState.IDLE:
                            self.behavior_manager.current_state = State.IDLE
                            self.behavior_manager._reset_state_duration()
                        elif self.agent_state == AgentState.TALKING:
                            self.behavior_manager.current_state = State.TALK
                            self.behavior_manager._reset_state_duration()
                        elif self.agent_state == AgentState.THINKING:
                            self.behavior_manager.current_state = State.THINKING
                            self.behavior_manager._reset_state_duration()
                        current_state = self.behavior_manager.get_current_state()
                
                # Check if animation restart is needed
                force_restart = self.behavior_manager._need_restart_animation
                if force_restart:
                    self.behavior_manager._need_restart_animation = False
                self.renderer.render(current_state, force_restart=force_restart)
                
                # Note: Subtitle updates are handled by output_loop(), no need to update here
        except Exception as e:
            print(f"Update loop error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_dialog_start_recording(self):
        """Dialog start recording button"""
        self.manual_recording = True
        
        # Switch to thinking state when starting recording
        self._safe_set_state(AgentState.THINKING, "thinking")
        
        # Update GUI using thread-safe method
        if self.voice_dialog:
            self.voice_dialog.update_status_safe("Status: Recording...")
            self.voice_dialog.result_text.clear()  # This is in main thread, can call directly
        
        # Start recording in background thread (use fixed duration, interrupt when user clicks stop)
        def record_audio():
            try:
                # Use longer recording duration, but will be interrupted when user clicks stop
                # Actually use listen_once, but set shorter timeout to respond to stop button
                result = self.voice_recognition.listen_once(
                    timeout=30.0,  # Maximum 30 seconds
                    phrase_time_limit=30.0  # Maximum 30 seconds
                )
                
                # Recording complete, process result (update GUI using thread-safe method)
                if not self.manual_recording:
                    # User manually stopped
                    if self.voice_dialog:
                        self.voice_dialog.update_status_safe("Status: Stopped")
                    # Return to idle state
                    self.agent_state = AgentState.IDLE
                    self.behavior_manager.set_next_state("idle")
                    return
                
                if result.is_success:
                    print(f"\n[Recording Thread] ✓ Voice recognition successful")
                    print(f"[Recording Thread] Recognition result: '{result.text}'")
                    print(f"[Recording Thread] Preparing to pass to main thread for processing...")
                    
                    if self.voice_dialog:
                        self.voice_dialog.update_result_safe(result.text)
                        self.voice_dialog.update_status_safe("Status: Recognition successful, processing...")
                    
                    # Keep thinking state, process voice recognition result in main thread
                    recognized_text = result.text
                    print(f"[Recording Thread] Preparing to pass text to main thread: '{recognized_text}'")
                    
                    # Prefer signal-slot mechanism (most reliable)
                    if self.voice_signal is not None:
                        print(f"[Recording Thread] Using signal-slot mechanism to pass text (recommended method)")
                        try:
                            self.voice_signal.recognized.emit(recognized_text)
                            print(f"[Recording Thread] ✓ Signal sent: '{recognized_text}'")
                        except Exception as e:
                            print(f"[Recording Thread] ✗ Signal send failed: {e}")
                            import traceback
                            traceback.print_exc()
                            # If signal send fails, try direct call
                            print(f"[Recording Thread] Fallback: Directly calling _on_voice_recognized...")
                            try:
                                self._on_voice_recognized(recognized_text)
                                print(f"[Recording Thread] ✓ Direct call successful")
                            except Exception as e2:
                                print(f"[Recording Thread] ✗ Direct call failed: {e2}")
                                import traceback
                                traceback.print_exc()
                    elif self.app and QTimer:
                        # Fallback: use QTimer.singleShot
                        print(f"[Recording Thread] Using QTimer.singleShot to pass (fallback method)")
                        print(f"[Recording Thread] app object: {self.app}")
                        print(f"[Recording Thread] QTimer class: {QTimer}")
                        
                        # Create a wrapper function to ensure callback execution
                        def callback_wrapper():
                            print(f"\n[Callback Wrapper] ========== QTimer callback triggered ==========")
                            print(f"[Callback Wrapper] Current thread: {threading.current_thread().name}")
                            print(f"[Callback Wrapper] Preparing to call _on_voice_recognized")
                            print(f"[Callback Wrapper] Text content: '{recognized_text}'")
                            try:
                                self._on_voice_recognized(recognized_text)
                                print(f"[Callback Wrapper] ✓ _on_voice_recognized call completed")
                            except Exception as e:
                                print(f"[Callback Wrapper] ✗ _on_voice_recognized call failed: {e}")
                                import traceback
                                traceback.print_exc()
                            print(f"[Callback Wrapper] ========================================\n")
                        
                        QTimer.singleShot(0, callback_wrapper)
                        print(f"[Recording Thread] ✓ QTimer.singleShot set")
                        
                        # Process events
                        if self.app:
                            try:
                                for i in range(10):
                                    self.app.processEvents()
                                    import time
                                    time.sleep(0.01)
                            except Exception as e:
                                print(f"[Recording Thread] ⚠ processEvents() failed: {e}")
                    else:
                        # If no GUI, call directly
                        print(f"[Recording Thread] No GUI, directly calling _on_voice_recognized")
                        try:
                            self._on_voice_recognized(result.text)
                            print(f"[Recording Thread] ✓ Direct call completed")
                        except Exception as e:
                            print(f"[Recording Thread] ✗ Direct call failed: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    if self.voice_dialog:
                        self.voice_dialog.update_status_safe(f"Status: Recognition failed - {result.error_message}")
                    self.manual_recording = False
                    # Recognition failed, return to idle
                    self.agent_state = AgentState.IDLE
                    self.behavior_manager.set_next_state("idle")
            except Exception as e:
                print(f"Recording error: {e}")
                if self.voice_dialog:
                    self.voice_dialog.update_status_safe(f"Status: Error - {str(e)}")
                self.manual_recording = False
                # On error, return to idle
                self.agent_state = AgentState.IDLE
                self.behavior_manager.set_next_state("idle")
        
        threading.Thread(target=record_audio, daemon=True).start()
    
    def _on_dialog_stop_recording(self):
        """Dialog stop recording button"""
        # Stop recording (by setting flag, recording thread will check)
        self.manual_recording = False
        if self.voice_dialog:
            self.voice_dialog.update_status_safe("Status: Stopping recording, processing...")
        # Note: Actual recording stop needs to wait for listen_once to complete or timeout
        # This just sets the flag to let the recording thread know the user wants to stop
    
    def start(self):
        """Start AI Agent"""
        if self.running:
            print("AI Agent is already running")
            return
        
        print("=" * 60)
        print("Starting AI Agent")
        print("=" * 60)
        
        # Load model
        print("\n[Init] Checking and loading Qwen model...")
        try:
            # Check if model is already downloaded
            if self.text_generator.is_model_downloaded():
                print("[Init] Model exists, loading directly...")
            else:
                print("[Init] Model not found, need to download...")
                print("[Info] First download may take a long time, please wait...")
            
            self.text_generator.load_model()
            print("[Init] Model loading completed")
            
            # Set text generator to memory coordinator if available
            if self.memory_coordinator:
                self.memory_coordinator.set_text_generator(self.text_generator)
                print("[Init] ✓ Text generator set to memory coordinator")
        except Exception as e:
            print(f"[Init] Model loading failed: {e}")
            print("Please ensure model is downloaded, or check network connection")
            import traceback
            traceback.print_exc()
            return
        
        # Adjust ambient noise
        print("\n[Init] Adjusting ambient noise...")
        self.voice_recognition.adjust_for_ambient_noise(duration=1.0)
        
        # Initialize GUI
        if self.use_gui:
            try:
                self._init_gui()
                # If GUI initialization fails, fall back to console mode
                if self.app is None:
                    print("Warning: GUI initialization failed, switching to console mode")
                    self.use_gui = False
            except Exception as e:
                print(f"Warning: GUI initialization exception: {e}")
                import traceback
                traceback.print_exc()
                self.use_gui = False
                self.app = None
        
        # Set initial state
        self.running = True
        self.agent_state = AgentState.IDLE
        self.behavior_manager.set_next_state("idle")
        
        # If using GUI, show voice input dialog
        if self.use_gui and self.voice_dialog:
            try:
                self.voice_dialog.show()
            except Exception as e:
                print(f"Warning: Unable to show voice input dialog: {e}")
                self.voice_dialog = None
        
        # Start listening thread (automatic listening in non-GUI mode)
        if not self.use_gui:
            self.stop_listening_event.clear()
            self.listening_thread = threading.Thread(target=self._listening_loop, daemon=True)
            self.listening_thread.start()
        
        print("\n[Start] AI Agent started")
        if self.use_gui and self.voice_dialog:
            print("[Info] Use voice input dialog for recording")
        else:
            print("[Info] Starting to listen for voice input...")
            print("[Info] Press Ctrl+C to stop")
        
        # Main loop
        if self.use_gui:
            # GUI mode: run Qt event loop
            try:
                exit_code = self.app.exec_()
                print(f"Qt event loop exited with code: {exit_code}")
                self.stop()
            except KeyboardInterrupt:
                print("\n\n[Stop] Received stop signal...")
                self.stop()
        else:
            # Console mode: traditional loop
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n[Stop] Received stop signal...")
                self.stop()
    
    def stop(self):
        """Stop AI Agent"""
        print("\n[Stop] Stopping AI Agent...")
        self.running = False
        self.stop_listening_event.set()
        
        # Persist memory on stop
        if self.memory_coordinator:
            try:
                self.memory_coordinator.persist()
                print("[Stop] Memory coordinator persisted (Replay + RAG)")
            except Exception as e:
                print(f"[Stop] Error persisting memory coordinator: {e}")
        elif self.replay_memory:
            try:
                self.replay_memory._save_session()
                print("[Stop] Replay session persisted")
            except Exception as e:
                print(f"[Stop] Error persisting replay: {e}")
        
        # Stop voice synthesis
        if self.voice_synthesis:
            self.voice_synthesis.stop()
        
        # Stop timer
        if self.update_timer:
            self.update_timer.stop()
        
        # Close dialogs
        if self.voice_dialog:
            self.voice_dialog.close()
        
        # Close windows
        if self.window:
            self.window.close()
        
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=2.0)
        
        print("[Stop] AI Agent stopped")


def check_live2d_setup():
    """Check Live2D setup"""
    print("=" * 60)
    print("Checking Live2D Setup")
    print("=" * 60)
    
    # Check PyQt5
    if PYQT5_AVAILABLE:
        print("✓ PyQt5 available")
    else:
        print("✗ PyQt5 not available")
        return False
    
    # Check Live2D module
    if LIVE2D_AVAILABLE:
        print("✓ Live2D module available")
    else:
        print("✗ Live2D module not available")
        return False
    
    # Check model paths
    from utils.path_config import get_model_paths
    possible_paths = get_model_paths()
    print(f"\nChecking {len(possible_paths)} possible model paths:")
    found = False
    for path in possible_paths:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {path}")
        if exists:
            found = True
            print(f"    Found model: {path}")
    
    if not found:
        print("\nWarning: No Live2D model files found")
        print("Please ensure model files exist in one of the above paths")
        return False
    
    return True


if __name__ == "__main__":
    import os
    
    # Check Live2D setup
    if not check_live2d_setup():
        print("\nWarning: Live2D setup check failed")
        print("Program will continue running, but Live2D features may be unavailable")
        response = input("\nContinue? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled")
            exit(0)
    
    # Create AI Agent
    agent = AIAgent(
        use_streaming=True,  # Enable streaming inference
        language="zh-CN",    # Default Chinese
        use_gui=True,        # Enable GUI
        model_path=None      # Auto-detect model path
    )
    
    # Start
    agent.start()
