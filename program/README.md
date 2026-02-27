# Reorganize - AI Desktop Pet

Reorganized and modularized version of the AI Desktop Pet application.

## Directory Structure

```
Reorganize/
├── core/           # Core modules (behavior management, state management)
├── ui/             # UI modules (subtitle window, voice input dialog, Live2D widget)
├── voice/          # Voice modules (recognition, synthesis)
├── ai/             # AI modules (text generation, tool management)
├── renderer/       # Renderer modules (Live2D renderer)
└── utils/          # Utility modules (path configuration)
```

## External Libraries Installation

### Required Libraries

Install all required libraries using pip:

```bash
pip install -r requirements.txt
```

### External Libraries (Clone from GitHub)

Some libraries need to be installed from source:

#### 1. Live2D Python Wrapper

The `live2d-py` library should be installed from PyPI:

```bash
pip install live2d-py
```

#### 2. Fish Speech (Optional)

If you want to use Fish Speech for TTS, you need to clone and install it:

**Location:** Clone to the project root or a separate directory

```bash
# Clone Fish Speech repository
git clone https://github.com/fishaudio/fish-speech.git

# Install dependencies
cd fish-speech
pip install -e .[stable]
# Or for CPU version:
# pip install -e .[cpu]
# Or for specific CUDA version:
# pip install -e .[cu126]  # or cu128, cu129
```

**Note:** Fish Speech can also be used in API mode without local installation.

#### 3. Live2D Model Files

Live2D model files should be placed in one of these locations (relative to project root):

**Standard Live2D SDK Samples:**
- `Live2D/Samples/Resources/Hiyori/Hiyori.model3.json`
- `Live2D/Samples/Resources/Haru/Haru.model3.json`
- `Live2D/Samples/Resources/Mao/Mao.model3.json`

**live2d-py Examples:**
- `Encapsulation_Live2D/live2d-py/Resources/v3/Haru/Haru.model3.json`
- `Encapsulation_Live2D/live2d-py/Resources/v3/Mao/Mao.model3.json`

The application will automatically search for model files in these locations.

**Note:** The project root is the `LiveRobot` directory (the directory containing `main`, `Live2D`, `Encapsulation_Live2D`, etc.).

## Running the Application

```bash
cd Reorganize
python main.py
```

## Module Dependencies

- **core**: No external dependencies (pure Python)
- **ui**: Requires PyQt5
- **voice**: Requires SpeechRecognition, pyaudio, edge-tts, pygame
- **ai**: Requires torch, transformers, bitsandbytes, accelerate
- **renderer**: Requires live2d-py, PyQt5
- **utils**: No external dependencies (pure Python)
