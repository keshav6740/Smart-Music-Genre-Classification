# music_genre_classifier/predict.py
import argparse
import os
import numpy as np
import librosa
from keras.models import load_model

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
SAMPLE_RATE = 22050
FIXED_TIME_FRAMES = 200


def extract_features_librosa(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < FIXED_TIME_FRAMES:
            pad_width = FIXED_TIME_FRAMES - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :FIXED_TIME_FRAMES]

        return mel_db[..., np.newaxis]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def predict(file_path, model):
    features = extract_features_librosa(file_path)
    if features is None:
        return None

    input_data = np.expand_dims(features, axis=0)
    predictions = model.predict(input_data, verbose=0)
    predicted_index = np.argmax(predictions)
    return GENRES[predicted_index]


def predict_folder(folder_path, model):
    results = {}
    for file in os.listdir(folder_path):
        if file.lower().endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            genre = predict(file_path, model)
            results[file] = genre
            print(f"{file} => {genre}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict music genre from audio file")
    parser.add_argument('--file', type=str, help='Path to a single audio file')
    parser.add_argument('--folder', type=str, help='Path to a folder of audio files')
    parser.add_argument('--model', type=str, default='music_genre_classifier.keras', help='Path to .keras model file')
    args = parser.parse_args()

    model = load_model(args.model)

    if args.file:
        genre = predict(args.file, model)
        if genre:
            print(f"Predicted genre: {genre}")
    elif args.folder:
        predict_folder(args.folder, model)
    else:
        print("Please provide either --file or --folder argument.")


if __name__ == '__main__':
    main()
