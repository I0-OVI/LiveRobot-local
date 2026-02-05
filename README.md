# Live Robot Design

This repository provides a simple and practical workflow for building a local Live2D-based AI desktop robot.
If you want to quickly try it out, just clone this repository, install the dependency and run main_live2D.py.

|Features                                       |Status|
|-----------------------------------------------|------|
|Live2D Character                               | ✅  |
|Speech recognition (Google API)                | ✅  |
|Test-to-Speech (online edge-tts)               | ✅  |
|AI communication(qwen model)                   | ✅  |
|Streaming text generation                      | ✅  |
|Personality Setting (system and dynamic prompt)| ✅  |
|Tools like weather searching                   | 🚧  |
|Momery (RAG, buffer ....)                      | 🚧  | 
|Emotion system                                 | 🚧  |
|Interaction of mouse or user's action          | 🚧  |

Notes
- ✅ means a feature that is currently available and working properly.
- 🚧 indicates a feature that is planned, under development, or requires further optimization.

## Acknowledgements

This project is inspired by Neuro-sama and AkagawaTsurunaki's video on bilibili.

This project is done by vibe coding(cursor).

This project makes use of the following open-source projects and resources:

- live2d-py (Python bindings for Live2D Cubism SDK)  
  https://github.com/Arkueid/live2d-py
  the main repository: https://github.com/AkagawaTsurunaki/ZerolanLiveRobot  

- Live2D Cubism SDK (Official SDK)  
  https://www.live2d.com  

- Qwen2 7B (4-bit quantized model, Hugging Face)  
  https://huggingface.co/Qwen  

Thanks to the authors and communities for their excellent work.
