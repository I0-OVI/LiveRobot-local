# Program (macOS) — AI Desktop Pet

This folder is the **macOS** build of the desktop pet app (sibling to `program/` for Windows). Paths resolve to the repository root (`LiveRobot-local/`) the same way: `Live2D/`, Hugging Face caches under `utils/models/`, etc.

## Prerequisites

- **Python 3.10+** recommended.
- **PyTorch with MPS** (Apple Silicon): install from [pytorch.org](https://pytorch.org) for your macOS version; CPU-only wheels work on Intel Macs (no MPS).
- **PyAudio**: requires PortAudio, e.g. `brew install portaudio` then `pip install pyaudio`.
- **bitsandbytes** / **turboquant**: not listed in `requirements.txt` — they target NVIDIA CUDA. On Mac, the LLM loads full precision (float16 on MPS when available, else float32 on CPU). Set `LIVEBOT_TURBOQUANT_BITS=0` if you copy env files from Windows.

## Install

```bash
cd program_mac
pip install -r requirements.txt
```

Install `live2d-py` with a wheel that includes **macOS** native libs (same PyPI package when a darwin build is published).

## Run

```bash
cd program_mac
python main.py
```

Memory (Replay + RAG): see `ai/memory/README.md` and `tools/README_RAG_tools.md`. Options in `setup.txt` under **`RAG_OPTIONS`**.

## Open app (keyword) — macOS

Phrases like **「帮我打开记事本」**, **「帮我打开 steam」**, **「open notepad」** use a whitelist-only launcher (**macOS**). Built-in targets map to native apps (TextEdit, Calculator, Preview, Finder, Terminal, Activity Monitor, Screenshot, System Settings, Edge, Chrome, Steam when installed under `/Applications`, etc.).

Optional **`open_app_allowlist.txt`** (next to `main.py`) adds **extra** apps by full path to a `.app` bundle or a Unix executable.

Search installed apps and append a line:

```bash
cd program_mac
python tools/app_path_lookup.py wechat
python tools/app_path_lookup.py WeChat --write --alias 微信
python tools/app_path_lookup.py steam --write --pick 2 --force
```

## Directory layout

Same as `program/README.md`: `core/`, `ui/`, `voice/`, `ai/`, `renderer/`, `utils/`.

## Model cache

Downloaded weights live under `utils/models/` (gitignored). To reuse the Windows `program/utils/models` cache without duplicating:

```bash
cd program_mac/utils
rm -rf models
ln -s ../../program/utils/models models
```

Create `utils/models` first if it does not exist.
