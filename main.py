
import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Set the WAV file path
wav_path = "/Users/panjiaguan/Desktop/audio/3/audio_converted.wav"

# Read audio with librosa
try:
    waveform, sample_rate = librosa.load(wav_path, sr=48000, mono=True)
    print(f" WAV Loading Success: sample_rate {sample_rate}")
except Exception as e:
    raise RuntimeError(f"WAV Loading Failure: {e}")

# Load the CLAP pre-training model
model_name = "laion/clap-htsat-unfused"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set audio window size
window_size = 5
slide_size = 2.5
num_samples_per_window = int(sample_rate * window_size)

# Calculate the CLAP audio embeddings for each window
embeddings = []
timestamps = []
for start in range(0, len(waveform) - num_samples_per_window, int(sample_rate * slide_size)):
    audio_segment = waveform[start:start + num_samples_per_window]

    inputs = processor(audios=audio_segment, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_audio_features(**inputs).detach().numpy()

    embeddings.append(embedding)
    timestamps.append(start / sample_rate)

embeddings = np.vstack(embeddings)
similarity_scores = cosine_similarity(embeddings)
change_scores = np.array([1 - similarity_scores[i, i+1] for i in range(len(embeddings)-1)])
threshold = np.mean(change_scores) + 1.5 * np.std(change_scores)
boundaries = [timestamps[i] for i in range(len(change_scores)) if change_scores[i] > threshold]

# Visualization split point
plt.figure(figsize=(10, 4))
plt.plot(timestamps[:-1], change_scores, label="Change Score")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
for boundary in boundaries:
    plt.axvline(x=boundary, color='g', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Change Score")
plt.title("Music Segmentation with CLAP")
plt.legend()
plt.show()

print("structural change point:", boundaries)

