"""
Built-in Hugging Face LLM presets (model id + local cache folder + load flags).
"""
from typing import Any, Dict, Optional, Tuple

# Preset id -> hub model name, cache subdir under program/utils/models/<subdir>, trust_remote_code for from_pretrained
LLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen3.5-4b": {
        "label": "Qwen3.5-4B (4-bit at load via bitsandbytes, ~4B)",
        # Official BF16/FP16 weights from Hub; quantized to 4-bit in RAM/VRAM at load.
        # Community "bnb-4bit"-only repos often use packed shapes that break standard from_pretrained.
        "model_name": "Qwen/Qwen3.5-4B",
        "cache_subdir": "qwen3.5-4b",
        "trust_remote_code": True,
    },
    "qwen2.5-7b": {
        "label": "Qwen2.5-7B-Instruct (4-bit, ~7B)",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "cache_subdir": "qwen2.5-7b-instruct",
        "trust_remote_code": False,
    },
    "qwen3.5-9b-text": {
        "label": "Qwen3.5-9B text-only (4-bit, ~9B; no vision tower)",
        "model_name": "principled-intelligence/Qwen3.5-9B-text-only",
        "cache_subdir": "qwen3.5-9b-text-only",
        "trust_remote_code": True,
    },
}


def list_preset_ids() -> Tuple[str, ...]:
    return tuple(sorted(LLM_PRESETS.keys()))


def resolve_llm_preset(
    llm_preset: str,
    model_name_override: Optional[str] = None,
    model_cache_dir_override: Optional[str] = None,
    current_dir_for_models: Optional[str] = None,
) -> Tuple[str, str, bool, str]:
    """
    Returns (model_name, cache_dir, trust_remote_code, preset_id_used).

    If model_name_override is set, use it and derive cache dir from override or slug;
    llm_preset is only used for logging in that case (preset_id_used = llm_preset).
    """
    if model_name_override:
        mn = model_name_override
        if model_cache_dir_override is not None:
            cache_dir = model_cache_dir_override
        elif current_dir_for_models:
            sub = mn.replace("/", "--").replace(":", "_")
            cache_dir = __import__("os").path.join(current_dir_for_models, "models", f"custom--{sub}")
        else:
            cache_dir = ""
        trc = False
        return mn, cache_dir, trc, llm_preset

    if llm_preset not in LLM_PRESETS:
        raise ValueError(
            f"Unknown llm_preset={llm_preset!r}. Choose one of: {', '.join(LLM_PRESETS.keys())}"
        )
    cfg = LLM_PRESETS[llm_preset]
    mn = cfg["model_name"]
    trc = bool(cfg.get("trust_remote_code", False))
    if model_cache_dir_override is not None:
        cache_dir = model_cache_dir_override
    else:
        if not current_dir_for_models:
            raise ValueError("current_dir_for_models required when model_cache_dir_override is None")
        import os

        cache_dir = os.path.join(current_dir_for_models, "models", cfg["cache_subdir"])
    return mn, cache_dir, trc, llm_preset
