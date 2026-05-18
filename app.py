import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

# --------------------------
# Load models
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

feature_extractor = Model(
    inputs=cnn_model.input,
    outputs=cnn_model.get_layer(
        "feature_layer"
    ).output
)

# --------------------------
# Streamlit UI
# --------------------------

st.title(
    "Phishing URL Detection System"
)

st.write(
    "Enhanced Random Forest using CNN-LSTM Feature Extraction"
)

url_input = st.text_input(
    "Enter URL"
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

        # Convert URL to sequence
        sequence = tokenizer.texts_to_sequences(
            [url_input]
        )

        padded = pad_sequences(
            sequence,
            maxlen=200
        )

        # Extract deep features
        features = feature_extractor.predict(
            padded
        )

        # RF prediction
        prediction = rf_model.predict(
            features
        )[0]

        # Probability
        probability = rf_model.predict_proba(
            features
        )[0]

        confidence = np.max(
            probability
        )

        # Display result
        if prediction == 1:

            st.error(
                f"⚠️ PHISHING URL\n\nConfidence: {confidence:.2%}"
            )

        else:

            st.success(
                f"✅ LEGITIMATE URL\n\nConfidence: {confidence:.2%}"
            )