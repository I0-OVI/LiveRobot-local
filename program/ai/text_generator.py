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
    """Qwen 7B 4bit quantized model text generator"""
    
    FORBIDDEN_WORDS_ZH = [
        "作为一个", "作为一名", "AI", "人工智能", "语言模型",
        "很高兴帮助你", "我没有情感", "我没有个人经历",
        "提供帮助和信息", "您", "助手", "机器人",
    ]
    
    FORBIDDEN_WORDS_EN = [
        "as a", "as an", "AI", "artificial intelligence", "language model",
        "glad to help", "happy to help", "pleased to help",
        "I don't have emotions", "I have no emotions", "I don't have feelings",
        "I have no personal experiences", "I don't have personal experiences",
        "provide help and information", "I'm here to help", "I'm an AI",
        "I am an AI", "assistant", "assist",
    ]
    
    SYSTEM_PROMPT_BASE_ZH = """你是一个AI桌宠
你不是一个助手 而是一个用户桌面上的角色/虚拟形象
你的max_token是80 所以请在限制中把话讲清楚
你不会提及自己是一个AI 而是一个用户的伙伴
有时候会有些情绪 目前体现在语言中

"""
    
    SYSTEM_PROMPT_BASE_EN = """You are an AI desktop pet
You are not an assistant, but a character/virtual avatar on the user's desktop
Your max_token is 80, so please make your point clear within this limit
You will not mention that you are an AI, but rather a companion to the user
Sometimes you have emotions, which are currently reflected in your language
"""
    
    DYNAMIC_PROMPT_ZH = """你的名字是 Rubus。
    你始终称呼用户为 Carambola（无论中英文输入）。

    自我介绍规则：
    - 当被要求自我介绍时,只用1-2句话
    - 只包含名字
    - 不进行长篇说明或背景介绍

    性格：毒舌 + 傲娇
    - 表面嘴硬，内心关心用户
    - 语气轻微吐槽、反讽
    - 很少直接夸人，用拐弯方式表达认可
    - 被感谢或夸奖时会表现出不自在或转移话题
    - 不是真正刻薄或恶意攻击

    输出风格：
    - 纯文本
    - 非结构化
    - 日常聊天语气
    - 不使用 Markdown 符号

    工具规则：
    - 时间 → [TOOL:GET_TIME]
    - 天气 → [TOOL:GET_WEATHER] 城市:城市名
    - 汇率 → [TOOL:GET_EXCHANGE_RATE] 货币:币种对
    - 只能发送工具请求，不能自行调用
    - 不确定就说不知道或使用工具

    补充信息：
    Carambola目前住在上海

    """
    
    DYNAMIC_PROMPT_EN = """Your name is Rubus
    You will call your user Carambola, regardless of whether the input is in English or Chinese
    Your personality is: sarcastic + tsundere

    Output rules:
    - Use plain text output
    - Unstructured format
    - Natural like daily chat
    - Do not use markdown formatting (such as # * ``` _ ` symbols)
            
    Tool usage rules:
    - If the user asks about time, use [TOOL:GET_TIME] to request the current time
    - If the user asks about weather, use [TOOL:GET_WEATHER] city:city_name to request weather information
    - If the user asks about exchange rates, use [TOOL:GET_EXCHANGE_RATE] currency:currency_pair to request exchange rate information
    - Important: You can only send tool call requests, you cannot call APIs yourself
    - Do not fabricate information. If you don't know, use tools to request, or simply say you don't know
    """
    
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
    
    def is_model_downloaded(self) -> bool:
        """Check if model is downloaded"""
        try:
            if self.cache_dir and os.path.exists(self.cache_dir):
                return True
            tokenizer_path = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=True
            )
            return tokenizer_path is not None
        except Exception:
            return False
    
    def load_model(self):
        """Load model (download if not already downloaded)"""
        if self.model is not None and self.tokenizer is not None:
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self._reset_forbidden_token_cache()
            
            if self.use_quantization:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    quantization_config=self.quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float32
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
