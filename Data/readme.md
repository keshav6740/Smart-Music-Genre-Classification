# 📁 GTZAN Dataset - Overview

The `Data/` folder contains the GTZAN dataset used for training and evaluation of the smart music genre classifier model.

---

## 📚 Contents

### 1. `genres_original/`
- Contains 10 subfolders:
  - `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`
- Each folder has 100 `.wav` files (30 seconds each)
- Used directly for training and feature extraction using `librosa`

### 2. `images_original/`
- Spectrogram `.png` images for each `.wav` file (not used in current training pipeline)
- Useful if you want to try image-based CNN models

### 3. `features_30_sec.csv` / `features_3_sec.csv`
- Precomputed features extracted from audio clips
- `30_sec`: 1 row per original file
- `3_sec`: Multiple chunks per song (more data, better for classical ML)

---

## 🧠 Usage in This Project
- `genres_original/` is used in `train.py` for feature extraction and model training.
- `images_original/` and `.csv` files are currently **not used**, but are provided for experimentation.

---

## 📦 Source
Originally sourced from Kaggle:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

---

## 📜 License
Dataset is shared under the terms specified by the Kaggle uploader. Please cite accordingly when using in research or deployment.
