import numpy as np
import wave
from scipy.io import wavfile
from scipy.signal import spectrogram
from keras.models import load_model
import os
from pydub import AudioSegment

model = load_model("music_genre_classifier.h5")

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
SAMPLE_RATE = 22050
FIXED_TIME_FRAMES = 200  

def is_valid_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            return True
    except wave.Error:
        return False

def convert_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        wav_path = file_path.replace(os.path.splitext(file_path)[1], "_converted.wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

def extract_features(file_path):
    try:
        sr, signal = wavfile.read(file_path)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        freqs, times, Sxx = spectrogram(signal, sr, nperseg=512)
        Sxx = np.log1p(Sxx)

        if Sxx.shape[1] < FIXED_TIME_FRAMES:
            pad_width = FIXED_TIME_FRAMES - Sxx.shape[1]
            Sxx = np.pad(Sxx, ((0, 0), (0, pad_width)), mode='constant')
        else:
            Sxx = Sxx[:, :FIXED_TIME_FRAMES]

        return Sxx
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def predict_genre(file_path):
    if not is_valid_wav(file_path):
        print("Invalid WAV format. Trying conversion...")
        file_path = convert_audio(file_path)
        if not file_path:
            return None

    features = extract_features(file_path)
    if features is None:
        print("Error extracting features.")
        return None

    features = features[np.newaxis, ..., np.newaxis]

    predictions = model.predict(features)
    genre_index = np.argmax(predictions)
    predicted_genre = GENRES[genre_index]
    
    print(f"Predicted Genre: {predicted_genre} 🎵")
    return predicted_genre

if __name__ == "__main__":
    audio_path = input("Enter the path to the audio file: ")
    predict_genre(audio_path)
