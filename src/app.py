import os
from dotenv import load_dotenv
import streamlit as st
from langchain.memory import ConversationBufferMemory

from utils import transcribe_audio, get_response_llm, play_text_to_speech, load_whisper

load_dotenv()

st.set_page_config(page_title="Voice Bot", layout="centered")
st.markdown('<h1 style="color: darkblue;">🎙️ Voice Bot</h1>', unsafe_allow_html=True)

model = load_whisper()
memory = ConversationBufferMemory(memory_key="chat_history")

# Upload a WAV audio file
uploaded_file = st.file_uploader("Upload a voice recording (.wav only)", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Transcribe audio
    text = transcribe_audio(model, temp_path)

    if text:
        st.markdown(f"<div style='background-color:#eee;padding:10px;border-radius:5px;'>👤 User: {text}</div>", unsafe_allow_html=True)

        # Get LLM response
        response_llm = get_response_llm(user_question=text, memory=memory)

        st.markdown(f"<div style='background-color:#dfe6e9;padding:10px;border-radius:5px;'>🤖 Bot: {response_llm}</div>", unsafe_allow_html=True)

        # Speak response
        play_text_to_speech(response_llm)

        os.remove(temp_path)
    else:
        st.warning("Could not understand the audio. Please try again.")
