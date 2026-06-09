# Phishing Detector

A Streamlit-based phishing URL detection model that combines CNN-LSTM feature extraction with a Random Forest classifier.

## Overview

This repository hosts a web app for detecting whether a given URL is phishing or legitimate.
The app uses a pre-trained CNN-LSTM model to extract deep URL features, then feeds those features into an enhanced Random Forest classifier for the final prediction.

## Features

- Simple Streamlit user interface
- Deep feature extraction using a CNN-LSTM model
- Final classification using an enhanced Random Forest model
- Confidence score shown with predictions

## Files

- `app.py` - Main Streamlit application
- `cnn_lstm_model.keras` - Pre-trained CNN-LSTM model used as a feature extractor
- `enhanced_rf.pkl` - Saved Random Forest classifier
- `tokenizer.pkl` - Saved tokenizer for URL preprocessing
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python runtime specification for deployment

## Requirements

- Python 3.11+ recommended
- `streamlit`
- `tensorflow==2.16.1`
- `scikit-learn`
- `joblib`
- `numpy`
- `pandas`

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the environment:

- PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

- Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Launch the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

## Usage

1. Enter the URL you want to test in the input field.
2. Click the `Detect` button.
3. The app displays whether the URL is predicted as phishing or legitimate, along with a confidence score.

## Notes

- Ensure `cnn_lstm_model.keras`, `enhanced_rf.pkl`, and `tokenizer.pkl` are present in the project root.
- The model and tokenizer are pre-trained and loaded at app startup.

## License

This repository does not include a license file. Add one if you want to clarify reuse permissions.
