
# 🎙️ VoiceBot — AI-Powered Conversational Agent

**VoiceBot** is an intelligent voice-enabled chatbot that leverages the power of **LangChain**, **GROQ API**, and **LLaMA LLM** to answer any natural language question. The bot uses speech-to-text (STT), processes the query with a large language model, and then responds with human-like speech using text-to-speech (TTS).

---

## 🚀 Features

* 🔗 **LangChain Framework**: Manages prompts, chains, tools, and memory for better context-aware conversations.
* 🧠 **GROQ + LLaMA LLM**: Fast and efficient LLM inference using Meta’s LLaMA models on GROQ's infrastructure.
* 🗣️ **Speech-to-Text (STT)**: Converts user’s voice input to text using tools like OpenAI Whisper or Google Speech API.
* 🔊 **Text-to-Speech (TTS)**: Responds with natural-sounding voice using tools like gTTS or pyttsx3.
* 🌐 **Web-Based Interface**: Simple web app for seamless interaction with the bot.

---

## 🧰 Tech Stack

* **LangChain**
* **GROQ API (LLaMA Model)**
* **Whisper / Google Speech Recognition** (STT)
* **gTTS / pyttsx3** (TTS)
* **Flask / Streamlit / Gradio** (Frontend Interface)
* **Python 3.10+**

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/voicebot-groq-llama.git
cd voicebot-groq-llama
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_optional_langchain_key
```

---

## 🧠 How It Works

1. **User speaks** into the microphone.
2. **Speech is transcribed** to text via Whisper or Google STT.
3. **LangChain** constructs a prompt and sends it to **GROQ API** using the **LLaMA model**.
4. The **LLM response** is received and converted to speech via **TTS**.
5. The bot **responds vocally** to the user.

---

## 🧪 Run the App

```bash
python app.py
```

Or if you're using Streamlit:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
voicebot/
│
├── app.py                # Main application logic
├── agents/               # LangChain agent setup
├── services/
│   ├── stt.py            # Speech-to-text logic
│   ├── tts.py            # Text-to-speech logic
│   └── llama_chain.py    # LangChain + GROQ setup
├── templates/            # (Optional) HTML for web interface
├── static/               # JS/CSS files
├── .env                  # Environment variables
└── requirements.txt
```

---

## 🧠 Prompt Engineering with LangChain

The app uses LangChain’s LLM wrappers with a custom prompt template that routes all queries through the LLaMA model via the GROQ endpoint:

```python
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)
```

---



---

## 🛡️ License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain)
* [GROQ](https://groq.com/)
* [Meta LLaMA](https://ai.meta.com/llama/)
* [OpenAI Whisper](https://github.com/openai/whisper)
* [gTTS](https://pypi.org/project/gTTS/)
* [Gradio / Streamlit](https://www.gradio.app/)

---




