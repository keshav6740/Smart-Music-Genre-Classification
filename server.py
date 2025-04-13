# app.py — Flask backend for Music Genre Wizard
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import librosa
from keras.models import load_model
from pydub import AudioSegment
import uuid

app = Flask(__name__)
# Enable CORS properly with more permissive settings
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], 
                            "allow_headers": ["Content-Type", "Authorization"]}})

MODEL_PATH = 'music_genre_classifier.keras'
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
SAMPLE_RATE = 22050
FIXED_TIME_FRAMES = 200
model = load_model(MODEL_PATH)


def convert_to_wav(original_path):
    try:
        audio = AudioSegment.from_file(original_path)
        wav_path = original_path.rsplit('.', 1)[0] + '_converted.wav'
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format='wav')
        return wav_path
    except Exception as e:
        print("Audio conversion error:", e)
        return None


def extract_features(file_path):
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
        print("Feature extraction error:", e)
        return None


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict_genre():
    print(f"Received request with method: {request.method}")
    
    if request.method == 'OPTIONS':
        print("Handling OPTIONS request")
        return '', 204  # No content needed for preflight response

    try:
        if 'audio' not in request.files:
            print("⚠️ No file found in request")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['audio']
        if not file.filename:
            print("⚠️ Empty filename")
            return jsonify({'error': 'Empty filename'}), 400

        print(f"📥 Processing file: {file.filename}")

        os.makedirs('temp', exist_ok=True)
        unique_id = str(uuid.uuid4())
        original_path = os.path.join('temp', unique_id + '_' + file.filename)
        file.save(original_path)
        print(f"✅ Saved to: {original_path}")

        if not original_path.lower().endswith('.wav'):
            converted_path = convert_to_wav(original_path)
            os.remove(original_path)
            if not converted_path:
                print("❌ Conversion failed.")
                return jsonify({'error': 'Unable to convert audio'}), 500
            path_to_process = converted_path
        else:
            path_to_process = original_path

        features = extract_features(path_to_process)
        os.remove(path_to_process)

        if features is None:
            print("❌ Feature extraction failed.")
            return jsonify({'error': 'Unable to process audio'}), 500

        input_data = np.expand_dims(features, axis=0)
        predictions = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(predictions)
        predicted_genre = GENRES[predicted_index]

        print(f"🎶 Predicted genre: {predicted_genre}")
        print("📤 Sending JSON:", {'genre': predicted_genre})
        
        # Create response with proper headers
        response = jsonify({'genre': predicted_genre, 'success': True})
        return response

    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500


if __name__ == '__main__':
    print("Starting server on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0')  # Allow external connections