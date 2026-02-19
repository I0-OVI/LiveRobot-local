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

### Memory
🚧
#### Short-term Memory(Replay)
When using AI assistants such as ChatGPT, the model reviews recent conversation history when generating new responses. For example, when you ask a follow-up question, the system implicitly reviews the latest dialogue to maintain coherence. This mechanism is commonly referred to as replay.

Replay is relatively straightforward to implement: it involves storing recent conversation turns and feeding them back into the model as context for the next generation. However, it is constrained by the model’s context window. Therefore, only a limited number of recent interactions can be retained, which distinguishes short-term memory (replay) from long-term memory systems.

#### Long-term Memory(RAG)