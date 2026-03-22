# Quick Setup Guide

Quick setup guide for installing dependencies and configuring system prompts.

## Project Structure

```
LiveRobot/
├── program/              # Main application directory
│   ├── ai/              # AI modules (text generation, tool management)
│   ├── core/            # Core modules (behavior management)
│   ├── renderer/        # Renderer modules (Live2D renderer)
│   ├── ui/              # UI modules
│   ├── voice/           # Voice modules
│   ├── utils/           # Utility modules
│   ├── main.py          # Main entry point
│   └── requirements.txt # Dependencies
├── Live2D/              # Live2D model files (optional)
└── Encapsulation_Live2D/# Live2D encapsulation (optional)
```

## Installation

### 1. Install Dependencies

```bash
cd program
pip install -r requirements.txt
pip install live2d-py
```

### 2. Download Resources

#### Qwen Model

The default model is downloaded from Hugging Face on first run (unless already cached). Hub weights are **not** pre-quantized; with a GPU the app loads them in **4-bit** using bitsandbytes.

- **Default repo**: [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Qwen2.5 **Instruct**; chat uses tokenizer `chat_template` + `generate`, not legacy `model.chat()`).
- **Default cache directory** (when using built-in defaults): `program/utils/models/qwen2.5-7b-instruct` (Hugging Face layout: `models--Qwen--Qwen2.5-7B-Instruct/snapshots/...`).
- **Organization page**: https://huggingface.co/Qwen

**Windows Task Manager vs real VRAM:** You may see **GPU0** (Intel UHD) and **GPU1** (NVIDIA). The large number on the Intel GPU is mostly **system RAM** used as *shared* graphics memory — not fast discrete VRAM. On the NVIDIA page, look at **Dedicated GPU memory** (e.g. **8 GB** on many RTX 4070 Laptop SKUs): that is what **CUDA / PyTorch** uses for model weights. **Shared GPU memory** there is also **system RAM** the driver can borrow when dedicated is full; it is **much slower** and not the same as having more VRAM.

**Other checkpoints:** Plan around **dedicated VRAM**. **7B + 4-bit** (default) usually fits **~8 GB** dedicated with reasonable context. **14B-class 4-bit** may need more headroom; confirm with `nvidia-smi`. Lighter options include **Qwen2.5-3B-Instruct**. Override via `AIAgent(..., model_name=...)` in `program/main.py` (must be an Instruct model with a `chat_template` in the tokenizer).

**Qwen3 (e.g. [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)):** Same code path as Qwen2.5; we call `apply_chat_template(..., enable_thinking=False)` when supported so output matches **non-thinking** Qwen3 behavior (aligned with Qwen2.5-style chat). Use **`transformers>=4.51.0`** for Qwen3 (older versions raise `KeyError: 'qwen3'`). **8B + 4-bit** is a bit heavier than 7B — verify on your GPU. MoE variants need enough VRAM for active experts at your quantization level.

#### Live2D SDK

Download the official Live2D Cubism SDK:

- **Official SDK**: https://www.live2d.com
- Extract the SDK to `Live2D/` directory in the project root
- The model files should be placed in `Live2D/Samples/Resources/`

#### Live2D Python Wrapper

The `live2d-py` library is available on GitHub:

- **GitHub Repository**: https://github.com/Arkueid/live2d-py
- **Original Repository**: https://github.com/AkagawaTsurunaki/ZerolanLiveRobot
- Install via pip: `pip install live2d-py`
- Or clone to `Encapsulation_Live2D/live2d-py/` for local development

### 3. Live2D Model Files

Place Live2D model files in one of the following locations (relative to project root):

- `Live2D/Samples/Resources/Hiyori/Hiyori.model3.json`
- `Live2D/Samples/Resources/Haru/Haru.model3.json`
- `Live2D/Samples/Resources/Mao/Mao.model3.json`
- `Encapsulation_Live2D/live2d-py/Resources/v3/Haru/Haru.model3.json`
- `Encapsulation_Live2D/live2d-py/Resources/v3/Mao/Mao.model3.json`

The application will automatically search these paths.

## Configuration

System prompts and forbidden words are configured in `program/ai/text_generator.py`:

### Forbidden Words (Lines 38-51)

```python
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
```

### System Prompts (Lines 53-117)

- **Base Prompt** (Lines 53-66): Defines the core identity and behavior of the AI desktop pet
- **Dynamic Prompt** (Lines 68-117): Defines character name, personality, and behavior rules

Edit these strings to customize the character.

## Quick Start

```bash
cd program
python main.py
```
