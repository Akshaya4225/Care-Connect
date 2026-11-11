# CARE CONNECT — AI Skin Image Classification

A Flask + TensorFlow/Keras educational project scaffold reconstructed from the project description in the user's resume.

## Important
This is a rebuilt implementation, not the original lost source code or trained model. It does not contain the original dataset or weights. The previous `85%+` figure must not be claimed for this rebuild unless your retrained model achieves it on a properly held-out test set. This is an educational image-classification demo, not a medical diagnostic tool.

## Features
- Responsive image-upload interface with preview and validation
- CNN training pipeline for folder-based image datasets
- Top-three class predictions with model confidence scores
- Nearby hospital/clinic search using OpenStreetMap Overpass data and browser geolocation
- Flask JSON API endpoints

## 1. Requirements
- Python 3.10 or 3.11 recommended
- A working internet connection for package installation and nearby facility search
- A compatible TensorFlow installation for your operating system

## 2. Create and activate a virtual environment (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.11 first or use another TensorFlow-supported Python version.

## 3. Add the dataset
Obtain a dataset you are permitted to use. Your earlier resume referenced `pacificrm/skindiseasedataset` on Kaggle, but verify that the dataset is still available, that its license permits your intended use, and that its labels are appropriate before using it.

Arrange images like this (the class folder names become labels):
```text
dataset/
  train/
    class_01/
      image1.jpg
    class_02/
      image2.jpg
    ... (22 class folders if using 22 classes)
  validation/
    class_01/
      image3.jpg
    class_02/
      image4.jpg
    ... (same 22 class folders)
```
Use a patient-level split where possible so images of the same person do not appear in both train and validation. Do not use the test set for tuning. Keep the class names and labels clinically appropriate and sourced from the dataset documentation.

## 4. Train the CNN
```powershell
python train.py
```
The script saves:
- `models/skin_disease_model.keras`
- `models/class_names.json`
- `models/training_report.json`

The printed validation accuracy is a validation metric, not a substitute for a separate held-out test evaluation. Record class-wise precision/recall and a confusion matrix before reporting performance.

## 5. Run the web application
```powershell
python app.py
```
Open `http://127.0.0.1:5000` in your browser. The prediction feature is enabled only after the model files have been generated. Hospital search requires location permission and an internet connection.

## API
- `GET /api/status` — model readiness
- `POST /api/predict` — multipart form field `image`
- `GET /api/hospitals?lat=<latitude>&lon=<longitude>` — nearby medical facilities

## Project structure
```text
CARE_CONNECT/
  app.py
  hospitals.py
  train.py
  requirements.txt
  templates/index.html
  static/style.css
  static/app.js
  models/
  dataset/                 # add your permitted dataset here
```

## Before submitting to an internship
1. Train the model using a properly split dataset.
2. Test it on images that were not used for training or validation.
3. Keep evidence of the actual test results and limitations.
4. Add your own screenshots and a short demo recording.
5. Explain which parts you implemented and which external dataset/API you used.
6. Do not upload private, identifiable patient images or dataset files without permission.
