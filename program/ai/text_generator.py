"""
Qwen Instruct models, 4-bit loading via bitsandbytes when CUDA is available.
Uses HuggingFace Transformers chat templates + generate; no legacy Qwen1 model.chat() / qwen_generation_utils.
"""
import os
import queue
import re
import threading
import torch
from concurrent.futures import Future
from typing import Optional, List, Tuple, Set
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    LogitsProcessor,
    LogitsProcessorList,
)
from threading import Thread
from .tool_manager import ToolManager
from utils.setup_loader import load_setup, get_user_name, parse_forbidden_words_list
from utils.time_context import local_time_context_block


class ForbiddenWordsLogitsProcessor(LogitsProcessor):
    """Forbidden words logits processor"""

    def __init__(self, forbidden_token_ids: Set[int], penalty: float = -float("inf")):
        self.forbidden_token_ids = forbidden_token_ids
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for token_id in self.forbidden_token_ids:
            if token_id < scores.size(-1):
                scores[:, token_id] = self.penalty
        return scores


class _SyncChatJob:
    """Runs _run_chat_direct on the inference worker thread."""

    __slots__ = ("gen", "user_input", "history", "enhanced_prompt", "future", "max_new_tokens")

    def __init__(self, gen, user_input, history, enhanced_prompt, future, max_new_tokens: int = 80):
        self.gen = gen
        self.user_input = user_input
        self.history = history
        self.enhanced_prompt = enhanced_prompt
        self.future = future
        self.max_new_tokens = max_new_tokens

    def execute(self) -> None:
        try:
            out = self.gen._run_chat_direct(
                self.user_input,
                self.history,
                self.enhanced_prompt,
                max_new_tokens=self.max_new_tokens,
            )
            self.future.set_result(out)
        except BaseException as e:
            self.future.set_exception(e)


class _StreamChatJob:
    """Runs streaming generation on the worker; caller reads token_queue."""

    __slots__ = ("gen", "user_input", "history", "enhanced_prompt", "out_q")

    def __init__(self, gen, user_input, history, enhanced_prompt, out_q):
        self.gen = gen
        self.user_input = user_input
        self.history = history
        self.enhanced_prompt = enhanced_prompt
        self.out_q = out_q

    def execute(self) -> None:
        try:
            for piece in self.gen._iter_chat_stream_direct(
                self.user_input, self.history, self.enhanced_prompt
            ):
                self.out_q.put(("chunk", piece))
            self.out_q.put(("done", None))
        except BaseException as e:
            self.out_q.put(("error", e))


class _WeatherNaturalizeJob:
    """Second-pass weather wording on the inference worker (avoids queue deadlock)."""

    __slots__ = ("gen", "user_input", "history", "facts", "future")

    def __init__(self, gen, user_input, history, facts, future):
        self.gen = gen
        self.user_input = user_input
        self.history = history
        self.facts = facts
        self.future = future

    def execute(self) -> None:
        try:
            text = self.gen._weather_naturalize_direct(self.user_input, self.history, self.facts)
            self.future.set_result(text)
        except BaseException as e:
            self.future.set_exception(e)


class QwenTextGenerator:
    """Qwen Instruct (default Qwen2.5-7B) 4-bit text generator. Prompts and forbidden words from setup.txt."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize Qwen text generator

        Args:
            model_name: HuggingFace model name, default "Qwen/Qwen2.5-7B-Instruct"
            cache_dir: Model cache directory, if None uses default HuggingFace cache
            trust_remote_code: Passed to from_pretrained (some checkpoints need True)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = torch.cuda.is_available()

        self._forbidden_token_ids: Optional[Set[int]] = None
        self.tool_manager = ToolManager()

        # Load system prompt, role and forbidden words from setup.txt (minimal fallbacks if section missing)
        setup = load_setup()
        user_name = get_user_name()
        self.SYSTEM_PROMPT_BASE_ZH = setup.get("SYSTEM_PROMPT_ZH") or "你是一个AI桌宠。\n"
        self.SYSTEM_PROMPT_BASE_EN = setup.get("SYSTEM_PROMPT_EN") or "You are an AI desktop pet.\n"
        role_zh = setup.get("ROLE_ZH") or ""
        role_en = setup.get("ROLE_EN") or ""
        self.DYNAMIC_PROMPT_ZH = role_zh.replace("{USER_NAME}", user_name)
        self.DYNAMIC_PROMPT_EN = role_en.replace("{USER_NAME}", user_name)
        self.FORBIDDEN_WORDS_ZH = parse_forbidden_words_list(setup["FORBIDDEN_WORDS_ZH"]) if setup.get("FORBIDDEN_WORDS_ZH") else []
        self.FORBIDDEN_WORDS_EN = parse_forbidden_words_list(setup["FORBIDDEN_WORDS_EN"]) if setup.get("FORBIDDEN_WORDS_EN") else []

        if self.use_quantization:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            self.quantization_config = None

        # Single worker + queue: all model.generate run on one thread (no cross-thread GPU calls)
        self._task_queue: queue.Queue = queue.Queue()
        self._inference_worker: Optional[threading.Thread] = None

    def _start_inference_worker(self) -> None:
        if self.model is None:
            return
        if self._inference_worker is not None and self._inference_worker.is_alive():
            return

        def loop():
            while True:
                job = self._task_queue.get()
                try:
                    job.execute()
                finally:
                    self._task_queue.task_done()

        t = threading.Thread(target=loop, daemon=True, name="qwen_inference_worker")
        self._inference_worker = t
        t.start()

    def _ensure_inference_worker(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded, please call load_model() first")
        self._start_inference_worker()

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
            System prompt string (includes local time-of-day for weather / scene consistency)
        """
        if enhanced_prompt:
            core = enhanced_prompt
        else:
            has_chinese = any("\u4e00" <= char <= "\u9fff" for char in user_input)
            if has_chinese:
                core = self.SYSTEM_PROMPT_BASE_ZH + self.DYNAMIC_PROMPT_ZH
            else:
                core = self.SYSTEM_PROMPT_BASE_EN + self.DYNAMIC_PROMPT_EN

        time_block = local_time_context_block(user_input)
        if not time_block:
            return core
        return core.rstrip() + "\n\n" + time_block

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

        markdown_symbols = ["#", "*", "`", "_", "```"]
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
            penalty=-float("inf"),
        )

        return LogitsProcessorList([forbidden_processor])

    def _reset_forbidden_token_cache(self):
        """Reset forbidden token ID cache"""
        self._forbidden_token_ids = None

    def _get_model_cache_path(self) -> Optional[str]:
        """Get the model-specific cache path (e.g. cache_dir/models--Qwen--Qwen2.5-7B-Instruct)."""
        if not self.cache_dir:
            return None
        model_slug = self.model_name.replace("/", "--")
        path = os.path.join(self.cache_dir, f"models--{model_slug}")
        return path if os.path.exists(path) else None

    def _model_input_device(self) -> torch.device:
        if self.model is None:
            return torch.device(self.device)
        return next(self.model.parameters()).device

    def _pad_token_id(self) -> int:
        if self.tokenizer is None:
            return 0
        if self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        return int(self.tokenizer.eos_token_id)

    def _messages_from_turns(
        self,
        user_input: str,
        history: Optional[List[Tuple[str, str]]],
        system_prompt: str,
    ) -> List[dict]:
        messages: List[dict] = []
        if system_prompt and str(system_prompt).strip():
            messages.append({"role": "system", "content": system_prompt})
        for turn in history or []:
            if not isinstance(turn, (list, tuple)) or len(turn) != 2:
                continue
            u, a = turn[0], turn[1]
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_input})
        return messages

    def _apply_chat_template_prompt(self, messages: List[dict]) -> str:
        """
        Render chat prompt. Qwen3 templates accept enable_thinking=False for plain assistant text
        (thinking mode off, per Qwen3 docs). Older chat templates omit that argument — fall back on TypeError.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        base_kw = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, **base_kw, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **base_kw)

    def _prepare_inputs_chat(
        self,
        user_input: str,
        history: Optional[List[Tuple[str, str]]],
        enhanced_prompt: Optional[str],
    ) -> dict:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise RuntimeError(
                f"Tokenizer has no chat_template; cannot run Instruct chat for {self.model_name!r}"
            )

        system_prompt = self._get_system_prompt(user_input, enhanced_prompt=enhanced_prompt)
        messages = self._messages_from_turns(user_input, history, system_prompt)
        prompt_text = self._apply_chat_template_prompt(messages)
        device = self._model_input_device()
        enc = self.tokenizer(prompt_text, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

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
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=True,
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
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code,
                local_files_only=use_local_only,
            )
            self._reset_forbidden_token_cache()

            if self.use_quantization:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    quantization_config=self.quantization_config,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=use_local_only,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float32,
                    local_files_only=use_local_only,
                )

            if self.model is not None:
                self._fix_stream_generator_compatibility()
                self._start_inference_worker()
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def _fix_stream_generator_compatibility(self):
        """Fix transformers_stream_generator compatibility issues"""
        if not hasattr(self.model, "_validate_model_class"):
            def _validate_model_class(self_model):
                return True

            self.model._validate_model_class = _validate_model_class.__get__(self.model, type(self.model))

    @staticmethod
    def naturalizable_weather_facts(facts: str) -> bool:
        """True if tool output is successful observation text (not an error prompt)."""
        s = (facts or "").strip()
        if not s:
            return False
        bad_starts = ("抱歉", "需要查询天气", "无法查询", "缺少必要的库")
        return not any(s.startswith(p) for p in bad_starts)

    def _weather_naturalize_user_message(self, user_input: str, facts: str) -> str:
        if re.search(r"[\u4e00-\u9fff]", user_input or ""):
            return (
                "【只输出你要对用户说的口语，一两句；不要工具标记、不要区县/片区地名、不要播音腔】\n"
                f"用户原话：{user_input}\n观测：{facts}\n"
                "可以说说适不适合出门。"
            )
        return (
            "[Output only 1–2 spoken sentences for the user. No tool markers, no district names.]\n"
            f"User said: {user_input}\nObservation: {facts}"
        )

    def _weather_naturalize_direct(
        self,
        user_input: str,
        history: Optional[List[Tuple[str, str]]],
        facts: str,
    ) -> str:
        """Runs on the inference worker thread only (second generate pass)."""
        inner = self._weather_naturalize_user_message(user_input, facts)
        inputs = self._prepare_inputs_chat(inner, history, None)
        logits_processor = self._create_logits_processor()
        gen_kwargs = {
            **inputs,
            "max_new_tokens": 128,
            "temperature": 0.62,
            "repetition_penalty": 1.12,
            "do_sample": True,
            "pad_token_id": self._pad_token_id(),
        }
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor
        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)
        inp_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, inp_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return self._filter_forbidden_words(text).strip()

    def _apply_weather_naturalization(
        self,
        user_input: str,
        history: Optional[List[Tuple[str, str]]],
        merged: str,
        trace: Optional[Tuple[str, str]],
    ) -> str:
        if not trace or trace[0] != "weather":
            return merged
        facts = trace[1]
        if not self.naturalizable_weather_facts(facts):
            return merged
        try:
            return self._weather_naturalize_direct(user_input, history, facts)
        except Exception:
            return merged

    def naturalize_weather_reply(
        self,
        user_input: str,
        weather_facts: str,
        history: Optional[List[Tuple[str, str]]] = None,
        timeout: float = 90.0,
    ) -> str:
        """
        Turn raw weather facts into a short spoken reply (for keyword tool path on UI thread).
        """
        self._ensure_inference_worker()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded, please call load_model() first")
        if not self.naturalizable_weather_facts(weather_facts):
            return weather_facts.strip()
        fut: Future = Future()
        self._task_queue.put(
            _WeatherNaturalizeJob(self, user_input, list(history or []), weather_facts, fut)
        )
        return fut.result(timeout=timeout).strip()

    def _run_chat_direct(
        self,
        user_input: str,
        history: Optional[List[Tuple[str, str]]] = None,
        enhanced_prompt: Optional[str] = None,
        max_new_tokens: int = 80,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Run generate on the current thread only. Used by the inference worker and stream fallback.
        Do not call from arbitrary threads; public API is chat().
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded, please call load_model() first")

        if history is None:
            history = []

        inputs = self._prepare_inputs_chat(user_input, history, enhanced_prompt)
        logits_processor = self._create_logits_processor()

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.3,
            "repetition_penalty": 1.15,
            "do_sample": True,
            "pad_token_id": self._pad_token_id(),
        }
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor

        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)

        inp_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, inp_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        response = self._filter_forbidden_words(response)
        response, _, trace = self.tool_manager.process_response(response)
        response = self._apply_weather_naturalization(user_input, history, response, trace)
        updated_history = list(history) + [(user_input, response)]
        return response, updated_history

    def chat(
        self,
        user_input: str,
        history: Optional[List[Tuple[str, str]]] = None,
        enhanced_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        max_new_tokens: int = 80,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Chat interface, returns reply and updated history.
        Dispatches to a single inference worker thread (queue).

        Args:
            user_input: User input text
            history: Conversation history
            enhanced_prompt: Optional enhanced prompt from RAG system
            timeout: If set, max seconds to wait for the worker (raises concurrent.futures.TimeoutError)
            max_new_tokens: Generation cap (default 80 for short pet replies; use higher for summaries, etc.)

        Returns:
            Tuple of (response, updated_history)
        """
        self._ensure_inference_worker()
        if history is None:
            history = []

        fut: Future = Future()
        self._task_queue.put(
            _SyncChatJob(
                self, user_input, list(history), enhanced_prompt, fut, max_new_tokens=max_new_tokens
            )
        )
        if timeout is not None:
            return fut.result(timeout=timeout)
        return fut.result()

    def chat_stream(self, user_input: str, history: Optional[List[Tuple[str, str]]] = None, enhanced_prompt: Optional[str] = None):
        """
        Stream chat interface, yields incremental text (worker produces tokens; this thread consumes).

        Args:
            user_input: User input text
            history: Conversation history
            enhanced_prompt: Optional enhanced prompt from RAG system

        Yields:
            Incremental response text
        """
        self._ensure_inference_worker()
        if history is None:
            history = []

        out_q: queue.Queue = queue.Queue()
        self._task_queue.put(
            _StreamChatJob(self, user_input, list(history), enhanced_prompt, out_q)
        )
        while True:
            msg = out_q.get()
            kind = msg[0]
            if kind == "chunk":
                yield msg[1]
            elif kind == "done":
                break
            elif kind == "error":
                try:
                    response, _ = self.chat(user_input, history, enhanced_prompt=enhanced_prompt)
                    yield response
                except Exception:
                    raise msg[1]
                break

    def _iter_chat_stream_direct(self, user_input: str, history: Optional[List[Tuple[str, str]]] = None, enhanced_prompt: Optional[str] = None):
        """
        Stream generation using TextIteratorStreamer (inference worker thread only).
        On failure, falls back via _run_chat_direct (never queues another job).

        Args:
            user_input: User input text
            history: Conversation history
            enhanced_prompt: Optional enhanced prompt from RAG system
        """
        try:
            inputs = self._prepare_inputs_chat(user_input, history, enhanced_prompt)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = {
                **inputs,
                "max_new_tokens": 80,
                "temperature": 0.3,
                "repetition_penalty": 1.15,
                "do_sample": True,
                "pad_token_id": self._pad_token_id(),
                "streamer": streamer,
            }
            logits_processor = self._create_logits_processor()
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor

            generation_thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
            generation_thread.start()

            full_response = ""
            for new_text in streamer:
                full_response += new_text
                filtered_response = self._filter_forbidden_words(full_response)
                yield filtered_response

            if full_response:
                base = full_response.strip()
                processed, _, trace = self.tool_manager.process_response(base)
                processed = self._apply_weather_naturalization(
                    user_input, history, processed, trace
                )
                if processed != base:
                    yield processed
        except Exception:
            response, _ = self._run_chat_direct(
                user_input, history, enhanced_prompt=enhanced_prompt, max_new_tokens=80
            )
            yield response
