# music_genre_classifier/train.py
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.models import Sequential
from keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

        return mel_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data(data_dir):
    features, labels = [], []
    for genre in GENRES:
        genre_dir = os.path.join(data_dir, genre)
        for file in os.listdir(genre_dir):
            if file.endswith('.wav'):
                path = os.path.join(genre_dir, file)
                feature = extract_features_librosa(path)
                if feature is not None:
                    features.append(feature)
                    labels.append(GENRES.index(genre))
    return np.array(features), np.array(labels)

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(GENRES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=GENRES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=GENRES, yticklabels=GENRES, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    data_dir = 'Data/genres_original'  # updated path to your dataset folder
    X, y = load_data(data_dir)
    X = X[..., np.newaxis]
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model((X.shape[1], X.shape[2], 1))
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ModelCheckpoint('music_genre_classifier.keras', monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(X_train, y_train, epochs=50, batch_size=64, 
                        validation_data=(X_test, y_test), callbacks=callbacks)

    plot_history(history)
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()