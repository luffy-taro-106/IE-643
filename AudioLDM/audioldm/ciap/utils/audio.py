import os
import librosa
import numpy as np

def load_audio(file_path, sr=22050):
    """Load an audio file."""
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def save_audio(file_path, audio, sr=22050):
    """Save an audio file."""
    librosa.output.write_wav(file_path, audio, sr=sr)

def audio_to_mel(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """Convert audio to mel spectrogram."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return mel

def mel_to_audio(mel, sr=22050, n_fft=2048, hop_length=512):
    """Convert mel spectrogram back to audio."""
    mel_db = librosa.power_to_db(mel, ref=np.max)
    audio = librosa.feature.inverse.mel_to_audio(mel_db, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return audio

def trim_audio(audio, top_db=20):
    """Trim silence from the beginning and end of an audio signal."""
    return librosa.effects.trim(audio, top_db=top_db)[0]