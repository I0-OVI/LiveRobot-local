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
🚧
**Input**

**Output**
