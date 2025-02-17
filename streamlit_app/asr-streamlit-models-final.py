import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration, WhisperProcessor, AutoProcessor, AutoModelForSpeechSeq2Seq
from pydub import AudioSegment
import numpy as np
import tempfile
import sounddevice as sd
import wave
from safetensors.torch import load_file
import os
import librosa

# Load the Wav2Vec2 processor and model
@st.cache_resource
def load_wav2vec2_model():
    processor = Wav2Vec2Processor.from_pretrained("../models/wav2vec2-large-xls-r-300m-hindi-total")
    model = Wav2Vec2ForCTC.from_pretrained("../models/wav2vec2-large-xls-r-300m-hindi-total")
    return processor, model
    
# Load the Whisper processor and model
@st.cache_resource
def load_whisper_model():
    processor = AutoProcessor.from_pretrained(r"../models/whisper-small-hindi-total")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(r"../models/whisper-small-hindi-total")
    return processor, model

# Function to preprocess audio
def preprocess_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(temp_wav.name, format="wav")
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    return samples

# Function to transcribe audio using Wav2Vec2
def transcribe_audio_wav2vec2(processor, model, input_values):
    inputs = processor(input_values, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# Function to transcribe audio using Whisper
def transcribe_audio_whisper(processor, model, audio):
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_features
    with torch.no_grad():
        # Generate using greedy decoding (simplest approach)
        # Explicitly set suppress_tokens to None
        predicted_ids = model.generate(input_values, suppress_tokens=None)

        # Decode predicted text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    st.write("Recording... Speak Now!")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())

    return temp_wav.name

# Streamlit App
def main():
    st.title("Hindi Speech-to-Text Transcription")
    st.write("Upload an audio file or record your voice for transcription.")

    # Dropdown for model selection    
    model_option = st.selectbox("Select Model", ["Whisper", "Wav2Vec2"])

    if model_option == "Whisper":
        processor, model = load_whisper_model()
    else:
        processor, model = load_wav2vec2_model()

    # Option to upload audio file
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])
    if st.button("Record Audio"):
        recorded_audio_path = record_audio()
        st.success("Recording complete!")
        audio_file = recorded_audio_path

    if audio_file is not None:
        if model_option == "Wav2Vec2":
            input_values = preprocess_audio(audio_file)
            transcription = transcribe_audio_wav2vec2(processor, model, input_values)
        else:
            audio, rate = librosa.load(audio_file, sr=16000)
            transcription = transcribe_audio_whisper(processor, model, audio)

        st.subheader("Transcription:")
        st.write(transcription)

if __name__ == "__main__":
    main()
