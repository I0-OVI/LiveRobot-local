"""
Qwen 7B 4bit quantized model text generation module
Uses HuggingFace transformers and bitsandbytes for 4bit quantization loading
"""
import os
import re
import torch
from typing import Optional, List, Tuple, Set
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TextIteratorStreamer,
    LogitsProcessor,
    LogitsProcessorList
)
from threading import Thread
from .tool_manager import ToolManager
from utils.setup_loader import load_setup, parse_forbidden_words_list


class ForbiddenWordsLogitsProcessor(LogitsProcessor):
    """Forbidden words logits processor"""
    
    def __init__(self, forbidden_token_ids: Set[int], penalty: float = -float('inf')):
        self.forbidden_token_ids = forbidden_token_ids
        self.penalty = penalty
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for token_id in self.forbidden_token_ids:
            if token_id < scores.size(-1):
                scores[:, token_id] = self.penalty
        return scores


class QwenTextGenerator:
    """Qwen 7B 4bit quantized model text generator. Prompts and forbidden words are loaded from setup.txt."""

    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat", cache_dir: Optional[str] = None):
        """
        Initialize Qwen text generator
        
        Args:
            model_name: HuggingFace model name, default "Qwen/Qwen-7B-Chat"
            cache_dir: Model cache directory, if None uses default HuggingFace cache
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = torch.cuda.is_available()
        
        self._forbidden_token_ids: Optional[Set[int]] = None
        self.tool_manager = ToolManager()

        # Load system prompt, role and forbidden words from setup.txt (minimal fallbacks if section missing)
        setup = load_setup()
        self.SYSTEM_PROMPT_BASE_ZH = setup.get("SYSTEM_PROMPT_ZH") or "你是一个AI桌宠。\n"
        self.SYSTEM_PROMPT_BASE_EN = setup.get("SYSTEM_PROMPT_EN") or "You are an AI desktop pet.\n"
        self.DYNAMIC_PROMPT_ZH = setup.get("ROLE_ZH") or ""
        self.DYNAMIC_PROMPT_EN = setup.get("ROLE_EN") or ""
        self.FORBIDDEN_WORDS_ZH = parse_forbidden_words_list(setup["FORBIDDEN_WORDS_ZH"]) if setup.get("FORBIDDEN_WORDS_ZH") else []
        self.FORBIDDEN_WORDS_EN = parse_forbidden_words_list(setup["FORBIDDEN_WORDS_EN"]) if setup.get("FORBIDDEN_WORDS_EN") else []

        if self.use_quantization:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            self.quantization_config = None
    
    def _filter_forbidden_words(self, text: str) -> str:
        """Filter forbidden words"""
        filtered_text = text
        for word in self.FORBIDDEN_WORDS_ZH + self.FORBIDDEN_WORDS_EN:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            filtered_text = pattern.sub("", filtered_text)
        filtered_text = " ".join(filtered_text.split())
        return filtered_text
    
    def _get_system_prompt(self, user_input: str = "", enhanced_prompt: Optional[str] = None) -> str:
        """
        Get system prompt based on user input language
        
        Args:
            user_input: User input text (for language detection)
            enhanced_prompt: Optional enhanced prompt from RAG (if provided, uses this instead)
        
        Returns:
            System prompt string
        """
        # If enhanced prompt is provided (from RAG), use it
        if enhanced_prompt:
            return enhanced_prompt
        
        # Otherwise, use default prompt based on language
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in user_input)
        
        if has_chinese:
            return self.SYSTEM_PROMPT_BASE_ZH + self.DYNAMIC_PROMPT_ZH
        else:
            return self.SYSTEM_PROMPT_BASE_EN + self.DYNAMIC_PROMPT_EN
    
    def _get_forbidden_token_ids(self) -> Set[int]:
        """Get all forbidden word token IDs"""
        if self._forbidden_token_ids is not None:
            return self._forbidden_token_ids
        
        if self.tokenizer is None:
            return set()
        
        forbidden_token_ids = set()
        all_forbidden_words = self.FORBIDDEN_WORDS_ZH + self.FORBIDDEN_WORDS_EN
        
        for word in all_forbidden_words:
            try:
                tokens = self.tokenizer.encode(word, add_special_tokens=False)
                forbidden_token_ids.update(tokens)
                tokens_with_space = self.tokenizer.encode(" " + word, add_special_tokens=False)
                forbidden_token_ids.update(tokens_with_space)
            except Exception:
                continue
        
        markdown_symbols = ['#', '*', '`', '_', '```']
        for symbol in markdown_symbols:
            try:
                tokens = self.tokenizer.encode(symbol, add_special_tokens=False)
                forbidden_token_ids.update(tokens)
                tokens_with_space_before = self.tokenizer.encode(" " + symbol, add_special_tokens=False)
                forbidden_token_ids.update(tokens_with_space_before)
                tokens_with_space_after = self.tokenizer.encode(symbol + " ", add_special_tokens=False)
                forbidden_token_ids.update(tokens_with_space_after)
                if len(symbol) > 1:
                    for char in symbol:
                        char_tokens = self.tokenizer.encode(char, add_special_tokens=False)
                        forbidden_token_ids.update(char_tokens)
            except Exception:
                continue
        
        self._forbidden_token_ids = forbidden_token_ids
        return forbidden_token_ids
    
    def _create_logits_processor(self) -> Optional[LogitsProcessorList]:
        """Create logits processor for blocking forbidden words"""
        if self.tokenizer is None:
            return None
        
        forbidden_token_ids = self._get_forbidden_token_ids()
        
        if not forbidden_token_ids:
            return None
        
        forbidden_processor = ForbiddenWordsLogitsProcessor(
            forbidden_token_ids=forbidden_token_ids,
            penalty=-float('inf')
        )
        
        return LogitsProcessorList([forbidden_processor])
    
    def _reset_forbidden_token_cache(self):
        """Reset forbidden token ID cache"""
        self._forbidden_token_ids = None
    
    def _get_model_cache_path(self) -> Optional[str]:
        """Get the model-specific cache path (e.g. cache_dir/models--Qwen--Qwen-7B-Chat)."""
        if not self.cache_dir:
            return None
        model_slug = self.model_name.replace("/", "--")
        path = os.path.join(self.cache_dir, f"models--{model_slug}")
        return path if os.path.exists(path) else None

    def is_model_downloaded(self) -> bool:
        """Check if model is fully downloaded (has snapshots and key files). Only then use local_files_only."""
        try:
            cache_path = self._get_model_cache_path()
            if cache_path:
                snapshots_path = os.path.join(cache_path, "snapshots")
                if os.path.isdir(snapshots_path):
                    snapshots = os.listdir(snapshots_path)
                    if snapshots:
                        snapshot_dir = os.path.join(snapshots_path, snapshots[0])
                        for key in ("config.json", "tokenizer_config.json"):
                            if os.path.isfile(os.path.join(snapshot_dir, key)):
                                return True
            try:
                AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
                return True
            except (OSError, ValueError):
                return False
        except Exception:
            return False
    
    def load_model(self):
        """Load model (download if not already downloaded). When already in cache, use local only to avoid hitting HuggingFace."""
        if self.model is not None and self.tokenizer is not None:
            return
        
        use_local_only = self.is_model_downloaded()
        if use_local_only:
            # Prevent any HuggingFace Hub request (e.g. custom_generate/generate.py) during/after load
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=use_local_only
            )
            self._reset_forbidden_token_cache()
            
            if self.use_quantization:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    quantization_config=self.quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=use_local_only
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    local_files_only=use_local_only
                )
            
            if self.model is not None:
                self._fix_stream_generator_compatibility()
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _fix_stream_generator_compatibility(self):
        """Fix transformers_stream_generator compatibility issues"""
        if not hasattr(self.model, '_validate_model_class'):
            def _validate_model_class(self_model):
                return True
            self.model._validate_model_class = _validate_model_class.__get__(self.model, type(self.model))
    
    def chat(self, user_input: str, history: Optional[List[Tuple[str, str]]] = None, enhanced_prompt: Optional[str] = None) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Chat interface, returns reply and updated history
        
        Args:
            user_input: User input text
            history: Conversation history
            enhanced_prompt: Optional enhanced prompt from RAG system
        
        Returns:
            Tuple of (response, updated_history)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded, please call load_model() first")
        
        if history is None:
            history = []
        
        from transformers import GenerationConfig
        
        generation_config = self.model.generation_config if hasattr(self.model, 'generation_config') else None
        
        if generation_config:
            new_config = GenerationConfig.from_dict(generation_config.to_dict())
            new_config.max_new_tokens = 80
            new_config.temperature = 0.3
            new_config.repetition_penalty = 1.15
        else:
            new_config = None
        
        system_prompt = self._get_system_prompt(user_input, enhanced_prompt=enhanced_prompt)
        logits_processor = self._create_logits_processor()
        
        chat_kwargs = {}
        if logits_processor is not None:
            chat_kwargs['logits_processor'] = logits_processor
        
        if new_config:
            response, updated_history = self.model.chat(
                self.tokenizer,
                user_input,
                history=history,
                system=system_prompt,
                generation_config=new_config,
                **chat_kwargs
            )
        else:
            response, updated_history = self.model.chat(
                self.tokenizer,
                user_input,
                history=history,
                system=system_prompt,
                max_new_tokens=80,
                temperature=0.3,
                repetition_penalty=1.15,
                **chat_kwargs
            )
        
        response = self._filter_forbidden_words(response)
        response, has_tool_call = self.tool_manager.process_response(response)
        
        return response, updated_history
    
    def chat_stream(self, user_input: str, history: Optional[List[Tuple[str, str]]] = None, enhanced_prompt: Optional[str] = None):
        """
        Stream chat interface, yields incremental text
        
        Args:
            user_input: User input text
            history: Conversation history
            enhanced_prompt: Optional enhanced prompt from RAG system
        
        Yields:
            Incremental response text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded, please call load_model() first")
        
        if history is None:
            history = []
        
        try:
            yield from self._chat_stream_with_streamer(user_input, history, enhanced_prompt=enhanced_prompt)
        except Exception:
            response, _ = self.chat(user_input, history, enhanced_prompt=enhanced_prompt)
            yield response
    
    def _chat_stream_with_streamer(self, user_input: str, history: Optional[List[Tuple[str, str]]] = None, enhanced_prompt: Optional[str] = None):
        """
        Stream generation using TextIteratorStreamer
        
        Args:
            user_input: User input text
            history: Conversation history
            enhanced_prompt: Optional enhanced prompt from RAG system
        """
        try:
            import sys
            import os
            
            model_path = None
            if hasattr(self.model, 'config') and hasattr(self.model.config, '_name_or_path'):
                model_name_or_path = self.model.config._name_or_path
                if self.cache_dir and os.path.exists(self.cache_dir):
                    for root, dirs, files in os.walk(self.cache_dir):
                        if 'qwen_generation_utils.py' in files:
                            model_path = root
                            break
            
            try:
                from qwen_generation_utils import make_context, get_stop_words_ids
                utils_available = True
            except ImportError:
                if model_path:
                    sys.path.insert(0, model_path)
                    try:
                        from qwen_generation_utils import make_context, get_stop_words_ids
                        utils_available = True
                    except ImportError:
                        utils_available = False
                else:
                    utils_available = False
            
            if not utils_available:
                raise ImportError("Cannot import qwen_generation_utils")
            
            system_prompt = self._get_system_prompt(user_input, enhanced_prompt=enhanced_prompt)
            generation_config = self.model.generation_config
            raw_text, context_tokens = make_context(
                self.tokenizer,
                user_input,
                history=history if history else [],
                system=system_prompt,
                max_window_size=getattr(generation_config, 'max_window_size', 6144),
                chat_format=getattr(generation_config, 'chat_format', 'chatml'),
            )
            
            stop_words_ids = get_stop_words_ids(
                getattr(generation_config, 'chat_format', 'chatml'), 
                self.tokenizer
            )
            
            input_ids = torch.tensor([context_tokens]).to(self.device)
            
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": 80,
                "temperature": 0.3,
                "repetition_penalty": 1.15,
                "stop_words_ids": stop_words_ids,
                "streamer": streamer,
            }
            
            logits_processor = self._create_logits_processor()
            if logits_processor is not None:
                generation_kwargs["logits_processor"] = logits_processor
            
            generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
            generation_thread.start()
            
            full_response = ""
            for new_text in streamer:
                full_response += new_text
                filtered_response = self._filter_forbidden_words(full_response)
                yield filtered_response
            
            if full_response:
                response, _ = self.tool_manager.process_response(full_response)
                if response != full_response:
                    yield response
        except Exception:
            response, _ = self.chat(user_input, history)
            yield response
