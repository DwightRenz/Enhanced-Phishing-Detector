import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
 
# --------------------------
# Load Models
# --------------------------
cnn_model = load_model(
    "cnn_lstm_model.keras"
)
rf_model = joblib.load(
    "enhanced_rf.pkl"
)
tokenizer = joblib.load(
    "tokenizer.pkl"
)
 
# --------------------------
# CNN-LSTM Feature Extractor
# --------------------------
cnn_model.build(
    input_shape=(None, 200)
)

# Feature extractor
feature_extractor = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.get_layer(
        "feature_layer"
    ).output
)
 
# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)
 
# --------------------------
# Title
# --------------------------
st.title(
    "🛡️ Phishing URL Detection System"
)
st.write(
    "Enhanced Random Forest using CNN-LSTM Feature Extraction"
)
st.markdown("---")
 
# --------------------------
# URL Input
# --------------------------
url_input = st.text_input(
    "Enter URL",
    placeholder="https://example.com"
)
 
# --------------------------
# Prediction
# --------------------------
if st.button("Detect"):
    if url_input.strip() == "":
        st.warning(
            "Please enter a URL"
        )
    else:
        # Convert URL to sequences
        sequence = tokenizer.texts_to_sequences(
            [url_input]
        )
 
        # Pad sequences
        padded = pad_sequences(
            sequence,
            maxlen=200
        )
 
        # Extract CNN-LSTM features
        features = feature_extractor.predict(
            padded,
            verbose=0
        )
 
        # Random Forest prediction
        prediction = rf_model.predict(
            features
        )[0]
 
        # Prediction probabilities
        probability = rf_model.predict_proba(
            features
        )[0]
 
        confidence = np.max(
            probability
        )
 
        st.markdown("---")
 
        # ✅ FIXED: prediction == 0 is PHISHING, prediction == 1 is LEGITIMATE
        if prediction == 0:
            st.error(
                f"⚠️ PHISHING URL DETECTED\n\nConfidence: {confidence:.2%}"
            )
        else:
            st.success(
                f"✅ LEGITIMATE URL\n\nConfidence: {confidence:.2%}"
            )
 
# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption(
    "Final Project: Enhanced Random Forest Classifier Using CNN-LSTM Feature Extraction for Real-Time Phishing URL Detection"
)
