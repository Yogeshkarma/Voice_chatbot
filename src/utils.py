import whisper
import os
from dotenv import load_dotenv

import wave
import pyaudio
from scipy.io import wavfile
import numpy as np

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from gtts import gTTS
import pygame
from io import BytesIO
import time


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    # Find the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=5):
    print("Recording...")
    frames = []
    # Calculate the number of chunks needed for the specified length of recording
    # 16000 Hertz -> sufficient for capturing the human voice
    # 1024 frames -> the higher, the higher the latency
    num_chunks = int(16000 / 1024 * chunk_length)

    # Record the audio data in chunks
    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = './temp_audio_chunk.wav'
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # Sample width
        wf.setframerate(16000)  # Sample rate
        wf.writeframes(b''.join(frames))  # Write audio frames

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")


def load_whisper():
    model = whisper.load_model("base")
    return model


def transcribe_audio(model, file_path):
    print("Transcribing...")
    # Print all files in the current directory
    print("Current directory files:", os.listdir())
    if os.path.isfile(file_path):
        results = model.transcribe(file_path) # , fp16=False
        return results['text']
    else:
        return None


def load_llm():
    chat_groq = ChatGroq(temperature=0, model_name="llama3-8b-8192",
                         groq_api_key=groq_api_key)
    return chat_groq


def get_response_llm(user_question, memory):
    # Initialize ChatGroq with the latest supported model
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"  # Updated model name
    )
    
    # Get chat history
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # Create simple context with chat history
    context = f"Previous conversation:\n{chat_history}\n\nHuman: {user_question}"
    
    # Get response
    response = llm.predict(context)
    
    # Save to memory
    memory.save_context({"input": user_question}, {"output": response})
    
    return response


def play_text_to_speech(text, speed=1.3):
    """
    Convert text to speech and play it with proper pygame initialization
    """
    try:
        # Initialize pygame and mixer
        pygame.init()
        pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=4096)
        
        # Create and save speech to BytesIO
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # Load and play
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error playing audio: {str(e)}")
        
    finally:
        # Clean up
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        pygame.quit()