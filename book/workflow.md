# Workflow

Even the AI assistant is very powerful, prompting it directly to make an AI live-robot without any detailed requirement is not a wise choice. The product tends to be unsatisfied.

### Reminder
To work efficiently with your version of the project, it is important to organize your files properly.  Create a folder structure like the following to separate code and resources:
```
--program/
---main.py
---render.py
---generation.py
--explanation files/
--github_dependencies/
requirements.txt
```
Proper file organization makes the project easier to manage and ensures that path dependencies are correct. It is recommended to set up this structure at the beginning of the project.


### State Machine

We designed several states, such as idle, thinking, and talk. Each state is associated with a time slice, which determines how long the character stays in that state. The user can manually input a state to trigger a change. If no input is provided, the states can loop automatically, e.g., `idle -> thinking -> talk`.

A state lock mechanism is implemented to prevent switching states before the current time slice expires. This is crucial because the Live2D rendering depends on it to know when to transition between animations smoothly.




### Live2D Deployment
In this project, we use the default Live2D character. The Live2D SDK is encapsulated in [Arkueid's repository](https://github.com/Arkueid/live2d-py). Before using this encapsulation, you need to download the official SDK files from [Live2D.com](https://www.live2d.com). Each character comes with multiple animations. It is recommended to record a description for each animation in a separate file since animations are indexed, for example 'Idle[4]' or 'TapBody[0]'. Without a proper description, it becomes difficult to assign animations to the 'correct' state. The descriptions do not need to be overly detailed; a simple note for each animation is enough, such as the following:
```
-Idle
--0: nod and look around
--1: blush and smile
--2: disappointed
-TapBody
--0：shake body
```
Next, import the `PyQt5` library to visualize the Live2D character. Each animation has a different duration, and the Live2D rendering is managed within the state machine. Locks are used to control the current stage, ensuring smooth transitions. It is important to note that the **animation lock and state lock are managed separately**.



### LLM Text Generation

Since I am very poor, I use open-source large language models (such as Qwen) instead of commercial APIs like ChatGPT or Gemini. These models also support streaming text generation. 

Streaming generation allows the system to start producing responses while the user is still speaking. For example, when the user says "Hello, how are you", the system can begin generating text as soon as it receives "Hello, how", rather than waiting for the complete sentence. This approach significantly reduces response latency and creates a more natural conversational experience. Using only a local LLM without streaming would not achieve the same level of real-time interaction.


### User Interface

#### Input
The system supports both speech recognition (Google API) and text input.  
In practice, the latency difference between local LLM text generation and speech streaming input is minimal. The time required for text input and speech streaming is approximately the same. 

This program would adjust the language mode based on input language. If the user input with English, the response will be English. If the user then input with Chinese, the model will response with Chinese.

#### Output
**Text to speech Edge-tts(online)**
Although there is a 0.5 to 1 second latency after text generation, this delay is within an acceptable range for the current stage of the project. Therefore, further optimization is postponed until other major components are completed.

The main challenge is **reducing the audible gap** between two adjacent sentences. A common approach is to place sentences into a queue and synthesize them one by one. However, this design introduces unavoidable silence between sentences, since the entire pipeline consists of: text input → synthesis → audio playback.

For example, without optimization, the sentence  
"Hello, how are you?"  
may be rendered as:  
"Hello ...... how are you"

Two strategies can be applied to alleviate this issue:
1. Pipelining: generate the next sentence while the current sentence is being played.  
2. Parallel synthesis: synthesize multiple sentences concurrently.

In practice, parallel synthesis provides better performance than pure pipelining, which still produces a small but noticeable gap. Therefore, this program enables two concurrent synthesis threads. To avoid playback order confusion, each synthesis task is indexed so that audio is played strictly in sequence.

**Limitation for edge-tts**: the LLM model could generate offline but this feature still requires vpn. This is because edge-tts is supported by Microsoft meaning prolong delay without vpn connection. 

I tried *piper* and *Fish speech* to convert this feature to offline but they didn't perform well as I expected. I will optimize this part in the future but with lower priority.

**Subtitle**
We use the `PyQt5` library to visualize the generated text. According to the length of each sentence, the subtitle frame automatically inserts line breaks or adjusts font size. Both English and Chinese are supported.


### Prompt
To personalize the agent, prompts are used to define both its identity and communication style.

**System Prompt** records the fundamental principles of text generation.
For example: `You are an AI desktop pet. You are not an assistant, but a character / virtual avatar on the user's desktop.`

This prompt mainly defines the role and high-level behavior of the agent. However, it is not suitable for fine-grained personality control, because its influence may gradually weaken as the conversation becomes longer.

**Dynamic Prompt** is used to define the character traits and speaking style, and is injected before every response is generated.

An example is shown below:
```
Personality: Sharp-tongued + Tsundere
- Outwardly stubborn, but genuinely cares about users
- Uses slightly sarcastic or witty tone
- Rarely praises directly, expresses approval indirectly
- Shows discomfort or changes the subject when thanked or praised
- Not truly mean or malicious
```
Since the dynamic prompt is re-applied for each generation, it provides more stable control over personality and speaking style during long conversations.
### Tools 
🚧
A dedicated function is implemented to determine whether a tool should be triggered based on the user’s input. The decision is made by detecting specific keywords, such as “what’s the weather” or “what time is it”, within the sentence. If the function returns True, the system then sends a structured request to the corresponding API to retrieve the required information.

It is important to emphasize that the API calls are handled entirely by the application layer rather than the language model itself. Although allowing the model to directly call external APIs may seem similar, it effectively grants the model control over system-level operations, which is difficult to manage and potentially unsafe. 

### Memory

The runtime flow is: **load short-term context → optionally retrieve long-term chunks → generate → append to replay → long-term write happens in the background.** `MemoryCoordinator` ties these steps together; finer options sit in `program/ai/memory` and `setup.txt` under `RAG_OPTIONS`.

#### Short-term memory (Replay)

When using AI assistants such as ChatGPT, the model reviews recent conversation history when generating new responses. For example, when you ask a follow-up question, the system implicitly reviews the latest dialogue to maintain coherence. This mechanism is commonly referred to as replay.

Replay is relatively straightforward to implement: it involves storing recent conversation turns and feeding them back into the model as context for the next generation. However, it is constrained by the model’s context window. Therefore, only a limited number of recent interactions can be retained, which distinguishes short-term memory (replay) from long-term memory systems.

#### Long-term memory (RAG)

**RAG testing**

Consider the two sentences 'My native language is Chinese.' and 'I always talk in Chinese.' At a quick glance, they may appear similar, since in many cases a person’s native language is also the one they use most often. Since embedding models capture structured *semantic differences* rather than relying on *intuition*, these two sentences may not be close in vector space, which means a query phrased in one way may fail to retrieve information written in the other form.

A similar issue appears with words like 'me' and 'user'. In a debugging context, they may refer to the same person, but semantically they are not equivalent. If the stored knowledge uses one form and the query uses the other, the system may not recognize them as related. To make retrieval more reliable, we need to normalize these variations before querying. This means mapping different surface expressions, such as pronouns and role labels, into a shared representation that matches the context of the application. This process is known as Query Canonicalization.

Long-term side uses a vector store and embeddings. Before each reply, a trigger decides whether to search (keywords, optional extra LLM pass, or always retrieve); results are merged into the prompt when relevant. After the reply, replay is updated right away; indexing into long-term storage is deferred (background worker or per-turn thread). Memory-related LLM calls and the main reply share a **single inference queue** so they run one after another on the GPU instead of in parallel.


### Thread management

The program is **multi-threaded by necessity**: the GUI must stay responsive, microphone I/O and recognition block, and the LLM is slow. The idea is to **keep Qt and Live2D on the main thread**, push blocking work to **short-lived or dedicated background threads**, and use **queues** where two threads exchange data so ordering stays explicit.

**Main thread (GUI mode)** runs the Qt event loop: window updates, subtitle widgets, dialog buttons, and (via `QTimer` / signals) safe updates to behavior state so Live2D does not fight with worker threads.

**Voice input** uses a **background thread** for recording and recognition; when text is ready, it hands the result back to the main thread through a **Qt signal** (or a timer callback fallback) so the next step does not block the UI. In **console-only** mode, a **listening loop thread** can run instead of the dialog.

**Each user turn** typically starts a **generation thread**: it loads memory context, calls **`chat_stream`** (which itself is fed by the inference worker—see below), fills a **`text_output_queue`**, and drives TTS from that thread as tokens arrive. A separate **output thread** **consumes** that queue to refresh subtitles and detect completion; finishing output returns the visible state toward **idle** on the main thread again. If a new turn starts while the old output is still running, the older output path is **forcibly torn down** first so queues and state do not accumulate garbage.

**Qwen inference** does not run from arbitrary threads. **`QwenTextGenerator`** keeps **one daemon worker** reading a **FIFO job queue**: every `chat` and every streaming job is executed there in order. Anything that needs the model—main reply, optional RAG trigger LLM, save-side LLM, etc.—**waits in line** behind that queue. That avoids undefined GPU concurrency; the trade-off is that a heavy background job can **delay** the next user-visible generation slightly.

**Long-term memory writes** run **after** the reply path: either a **single RAG save worker** draining its own queue, or **one daemon thread per save**, depending on configuration. That is separate from the Qwen queue but still **feeds** into it when those paths call `chat`.

**TTS** is discussed under **User Interface → Output**: parallel synthesis threads may exist for smoother playback; they are **indexed** so audio still plays in order.

**Miscellaneous** small daemons (e.g. waiting on a sound effect then hiding an overlay) follow the same pattern: **short background thread + signal or timer back to Qt**, without touching the model.


### Anon's Laughing and Crying
There are many related videos on Bilibili. Since we only need the sound effects from these videos, we extract the audio directly from the video links. We use the `yt-dlp` library to download and extract the audio from a given URL, which generates an `.m4a` audio file.

Open your terminal and run the following command:

```cmd
yt-dlp -f bestaudio "https://www.bilibili.com/video/BV...."
```

Here, `BV....` represents the BV number of the video.

The extracted audio file is usually in `.m4a` format. However, we cannot directly use this format in our system, so we convert it to `.wav` format using `ffmpeg`.

You can download `ffmpeg` from the following website: https://ffmpeg.org/download.html

**Convert `.m4a` to `.wav`**
```cmd
ffmpeg -i input.m4a output.wav
```
This command converts the `.m4a` audio file into `.wav` format.

**Crop the Audio**
```cmd
ffmpeg -i input.wav -ss 00:00:02 -to 00:00:05 -c copy output.wav
```
This command extracts the audio segment from 2 seconds to 5 seconds of the original audio file.

Before using `ffmpeg` directly in the terminal, we need to add it to the system `PATH`.  
Otherwise, we must use the `cd` command to navigate to the directory containing the `.exe` file before running it.

