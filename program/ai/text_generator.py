"""
Qwen Instruct models, 4-bit loading via bitsandbytes when CUDA is available.
Uses HuggingFace Transformers chat templates + generate; no legacy Qwen1 model.chat() / qwen_generation_utils.
"""
import json
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
    PreTrainedTokenizerFast,
)
from threading import Thread
from .tool_manager import ToolManager
from utils.setup_loader import load_setup, get_user_name, parse_forbidden_words_list
from utils.time_context import local_time_context_block


def _patch_bitsandbytes_params4bit_for_accelerate() -> None:
    """
    accelerate's set_module_tensor_to_device rebuilds 4-bit weights with
    Params4bit(new_value, requires_grad=..., **old_param.__dict__). PyTorch 2.x
    Parameter dict can include _is_hf_initialized, which bitsandbytes Params4bit.__new__
    does not accept. Forward only the kwargs the constructor supports.
    """
    try:
        import bitsandbytes.nn.modules as bnb_modules
    except ImportError:
        return
    P = getattr(bnb_modules, "Params4bit", None)
    if P is None or getattr(P, "_livebot_new_patched", False):
        return
    _orig_new = P.__new__

    def _new(cls, data=None, requires_grad=False, **kwargs):
        return _orig_new(
            cls,
            data,
            requires_grad,
            kwargs.get("quant_state"),
            kwargs.get("blocksize"),
            kwargs.get("compress_statistics", True),
            kwargs.get("quant_type", "fp4"),
            kwargs.get("quant_storage", torch.uint8),
            kwargs.get("module"),
            kwargs.get("bnb_quantized", False),
        )

    P.__new__ = _new  # type: ignore[method-assign]
    P._livebot_new_patched = True


def _patch_accelerate_execution_hook_for_bnb_meta() -> None:
    """
    accelerate.hooks.attach_execution_device_hook uses len(module.state_dict()) > 0.
    For bitsandbytes Linear4bit during GPU/CPU dispatch, state_dict() can call
    quant_state.as_dict() while tensors are still on meta, raising RuntimeError.
    Fall back to counting parameters/buffers without building a full state_dict.
    """
    try:
        import accelerate.hooks as acc_hooks
    except ImportError:
        return
    if getattr(acc_hooks, "_livebot_exec_hook_meta_patch", False):
        return

    def _module_has_tensors(module: torch.nn.Module) -> bool:
        try:
            return len(module.state_dict()) > 0
        except RuntimeError as e:
            msg = str(e).lower()
            if "meta" not in msg and "item()" not in msg:
                raise
            return any(True for _ in module.parameters(recurse=True)) or any(
                True for _ in module.buffers(recurse=True)
            )

    _orig = acc_hooks.attach_execution_device_hook

    def attach_execution_device_hook(
        module,
        execution_device,
        skip_keys=None,
        preload_module_classes=None,
        tied_params_map=None,
    ):
        if not hasattr(module, "_hf_hook") and _module_has_tensors(module):
            acc_hooks.add_hook_to_module(
                module,
                acc_hooks.AlignDevicesHook(
                    execution_device, skip_keys=skip_keys, tied_params_map=tied_params_map
                ),
            )

        if preload_module_classes is not None and module.__class__.__name__ in preload_module_classes:
            return

        for child in module.children():
            attach_execution_device_hook(
                child,
                execution_device,
                skip_keys=skip_keys,
                preload_module_classes=preload_module_classes,
                tied_params_map=tied_params_map,
            )

    acc_hooks.attach_execution_device_hook = attach_execution_device_hook  # type: ignore[assignment]
    acc_hooks._livebot_exec_hook_meta_patch = True


def _patch_accelerate_align_devices_hook_unwrap_forward() -> None:
    """
    AlignDevicesHook.pre_forward/post_forward are wrapped with torch.compiler.disable.
    That path can interact badly with bitsandbytes Params4bit moves during offload.
    Use the underlying callables (inspect.unwrap) so hooks stay eager.
    """
    try:
        import inspect
        import accelerate.hooks as acc_hooks
    except ImportError:
        return
    if getattr(acc_hooks, "_livebot_align_hook_unwrap", False):
        return
    ah = acc_hooks.AlignDevicesHook
    for name in ("pre_forward", "post_forward"):
        raw = ah.__dict__.get(name)
        if raw is None:
            continue
        inner = inspect.unwrap(raw)
        if inner is not raw:
            setattr(ah, name, inner)
    acc_hooks._livebot_align_hook_unwrap = True


_patch_bitsandbytes_params4bit_for_accelerate()
_patch_accelerate_execution_hook_for_bnb_meta()
_patch_accelerate_align_devices_hook_unwrap_forward()


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

        # Single worker + queue: all model.generate run on one thread (no cross-thread GPU calls)
        self._task_queue: queue.Queue = queue.Queue()
        self._inference_worker: Optional[threading.Thread] = None

    @staticmethod
    def _bnb_config(cpu_offload: bool) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
        )

    def _load_quantized_model(self, use_local_only: bool) -> None:
        """Prefer full GPU (device_map {0}); split + disk offload only if OOM or LIVEBOT_LLM_SPLIT=1."""
        base_kw = dict(
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            local_files_only=use_local_only,
        )
        force_split = os.environ.get("LIVEBOT_LLM_SPLIT", "").strip().lower() in ("1", "true", "yes")
        offload_dir = os.path.join(self.cache_dir or ".", "_hf_offload")
        os.makedirs(offload_dir, exist_ok=True)

        if not force_split and torch.cuda.is_available():
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=self._bnb_config(cpu_offload=False),
                    device_map={"": 0},
                    **base_kw,
                )
                return
            except Exception as e:
                oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
                if not oom:
                    raise
                torch.cuda.empty_cache()
                print(
                    "[Init] 4-bit model did not fit in GPU memory; retrying with auto device_map + offload "
                    f"(offload dir: {offload_dir})"
                )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self._bnb_config(cpu_offload=True),
            device_map="auto",
            offload_folder=offload_dir,
            **base_kw,
        )

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

    def _get_snapshot_dir(self) -> Optional[str]:
        """First revision under snapshots/ (HF hub cache layout)."""
        cache_path = self._get_model_cache_path()
        if not cache_path:
            return None
        snapshots_path = os.path.join(cache_path, "snapshots")
        if not os.path.isdir(snapshots_path):
            return None
        snapshots = sorted(os.listdir(snapshots_path))
        if not snapshots:
            return None
        return os.path.join(snapshots_path, snapshots[0])

    @staticmethod
    def _snapshot_has_model_weights(snapshot_dir: str) -> bool:
        if not os.path.isdir(snapshot_dir):
            return False
        for name in os.listdir(snapshot_dir):
            lower = name.lower()
            if lower.endswith(".safetensors") or lower == "pytorch_model.bin":
                return True
            if lower == "model.safetensors.index.json":
                return True
        return False

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
        """
        True only when the hub snapshot looks complete: config, weights, and tokenizer files.
        Avoids HF_HUB_OFFLINE + local_files_only when only tokenizer_config.json (or metadata) exists.
        """
        try:
            snap = self._get_snapshot_dir()
            if not snap:
                return False
            if not os.path.isfile(os.path.join(snap, "config.json")):
                return False
            if not self._snapshot_has_model_weights(snap):
                return False
            tok_ok = os.path.isfile(os.path.join(snap, "tokenizer.json")) or os.path.isfile(
                os.path.join(snap, "tokenizer.model")
            )
            return tok_ok
        except Exception:
            return False

    def _load_tokenizer_via_tokenizer_json_object(self, use_local_only: bool) -> PreTrainedTokenizerFast:
        """
        Qwen3.x repos often ship tokenizer_class=TokenizersBackend and tiktoken-backed metadata.
        AutoTokenizer / PreTrainedTokenizerFast.from_pretrained may try a broken slow->fast path.
        Loading tokenizer.json via the `tokenizers` library + tokenizer_object avoids that.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from tokenizers import Tokenizer as TokenizersTokenizer

        repo = self.model_name
        cache = self.cache_dir

        snap = self._get_snapshot_dir()
        tok_path = os.path.join(snap, "tokenizer.json") if snap else ""
        cfg_path = os.path.join(snap, "tokenizer_config.json") if snap else ""
        if not (snap and os.path.isfile(tok_path) and os.path.isfile(cfg_path)):
            try:
                tok_path = hf_hub_download(
                    repo_id=repo,
                    filename="tokenizer.json",
                    cache_dir=cache,
                    local_files_only=use_local_only,
                )
                cfg_path = hf_hub_download(
                    repo_id=repo,
                    filename="tokenizer_config.json",
                    cache_dir=cache,
                    local_files_only=use_local_only,
                )
            except LocalEntryNotFoundError:
                if not use_local_only:
                    raise
                tok_path = hf_hub_download(
                    repo_id=repo,
                    filename="tokenizer.json",
                    cache_dir=cache,
                    local_files_only=False,
                )
                cfg_path = hf_hub_download(
                    repo_id=repo,
                    filename="tokenizer_config.json",
                    cache_dir=cache,
                    local_files_only=False,
                )
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        fast = TokenizersTokenizer.from_file(tok_path)

        exclude = {
            "tokenizer_class",
            "auto_map",
            "backend",
            "chat_template",
            "chat_template_jinja",
        }
        init_kwargs = {k: v for k, v in cfg.items() if k not in exclude}
        try:
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=fast, **init_kwargs)
        except (TypeError, ValueError):
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=fast,
                eos_token=cfg.get("eos_token"),
                pad_token=cfg.get("pad_token"),
                bos_token=cfg.get("bos_token"),
                unk_token=cfg.get("unk_token"),
                model_max_length=cfg.get("model_max_length", 262144),
                clean_up_tokenization_spaces=cfg.get("clean_up_tokenization_spaces", False),
                add_prefix_space=cfg.get("add_prefix_space", False),
                errors=cfg.get("errors", "replace"),
                split_special_tokens=cfg.get("split_special_tokens", False),
            )

        tpl_path = ""
        if snap:
            cand = os.path.join(snap, "chat_template.jinja")
            if os.path.isfile(cand):
                tpl_path = cand
        try:
            if not tpl_path:
                tpl_path = hf_hub_download(
                    repo_id=repo,
                    filename="chat_template.jinja",
                    cache_dir=cache,
                    local_files_only=use_local_only,
                )
        except LocalEntryNotFoundError:
            if use_local_only:
                try:
                    tpl_path = hf_hub_download(
                        repo_id=repo,
                        filename="chat_template.jinja",
                        cache_dir=cache,
                        local_files_only=False,
                    )
                except Exception:
                    tpl_path = ""
            else:
                tpl_path = ""
        except Exception:
            tpl_path = ""

        if tpl_path and os.path.isfile(tpl_path):
            with open(tpl_path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = f.read()
        else:
            tpl = cfg.get("chat_template")
            if isinstance(tpl, str) and tpl.strip():
                tokenizer.chat_template = tpl

        return tokenizer

    def _load_tokenizer(self, use_local_only: bool):
        """
        Some Qwen3.x hubs set tokenizer_class to TokenizersBackend, which AutoTokenizer
        cannot resolve. Fall back to tokenizer.json + PreTrainedTokenizerFast(tokenizer_object=...).
        """
        kwargs = dict(
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            local_files_only=use_local_only,
        )
        try:
            return AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        except ValueError as e:
            err = str(e)
            if "TokenizersBackend" not in err and "TokenizersBackendFast" not in err:
                raise
            return self._load_tokenizer_via_tokenizer_json_object(use_local_only)

    def load_model(self):
        """Load model (download if not already downloaded). When already in cache, use local only to avoid hitting HuggingFace."""
        if self.model is not None and self.tokenizer is not None:
            return

        use_local_only = self.is_model_downloaded()
        prev_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        if use_local_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self.tokenizer = self._load_tokenizer(use_local_only)
            self._reset_forbidden_token_cache()

            if self.use_quantization:
                self._load_quantized_model(use_local_only)
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
        finally:
            if prev_hub_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_hub_offline

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
