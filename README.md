# Live Robot Design

This repository provides a simple and practical workflow for building a local Live2D-based AI desktop robot.
If you want to quickly try it out, just clone this repository, install the dependency and run main_live2D.py.

|Features                                       |Status|
|-----------------------------------------------|------|
|Live2D Character                               | ✅  |
|Speech recognition (Google API)                | ✅  |
|Test-to-Speech (Online Edge-tts)               | ✅  |
|AI communication(Qwen model)                   | ✅  |
|Streaming text generation                      | ✅  |
|Personality Setting (System and Dynamic prompt)| ✅  |
|Anon's Laughing and Crying                     | ✅  |
|Tools like weather searching                   | 🚧  |
|Short-term Memory (Replay)                     | 🚧  | 
|Long-term Memory (RAG)                         | 🚧  | 
|Emotion system                                 | 🚧  |
|Interaction of mouse or user's action          | 🚧  |

Notes
- ✅ means a feature that is currently available and working properly.
- 🚧 indicates a feature that is planned, under development, or requires further optimization.

PS.

Anon is Anon Chihaya. Anon Chihaya is a fictional character from the anime BanG Dream! It's MyGO!!!!!, part of the BanG Dream! franchise. She is the guitarist of the band MyGO!!!!!.
<p align ='center'>
<img src="./picture/爱音大笑.png" width="80">
<img src="./picture/爱音斯坦.jpg" width="70">


## Limitations
This subject is supported by open weights model. Even if we implement different prompt and add the memory, the answer is still stiff meaning differ from the normal speech. Perhaps the better model or having training could relieve this.

## Acknowledgements

This project is inspired by Neuro-sama and AkagawaTsurunaki's video on bilibili.

This project is done by vibe coding(cursor).

This project makes use of the following open-source projects and resources:

- live2d-py (Python bindings for Live2D Cubism SDK)  
  https://github.com/Arkueid/live2d-py
  the main repository: https://github.com/AkagawaTsurunaki/ZerolanLiveRobot  

- Live2D Cubism SDK (Official SDK)  
  https://www.live2d.com  

- Chat LLM: on startup a **GUI dialog** lets you pick **Qwen3.5-4B**, **Qwen2.5-7B-Instruct**, or **Qwen3.5-9B text-only** ([`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), [`principled-intelligence/Qwen3.5-9B-text-only`](https://huggingface.co/principled-intelligence/Qwen3.5-9B-text-only)); Hub ships full-precision weights; with CUDA the app loads them in **4-bit** via bitsandbytes (smaller VRAM than FP16/BF16). Skip the dialog with `python main.py --llm qwen3.5-4b` / `qwen2.5-7b` / `qwen3.5-9b-text`, or set env `LIVEBOT_LLM` to one of those ids. Force the dialog with `--llm gui`.  
  https://huggingface.co/Qwen  

Thanks to the authors and communities for their excellent work.
