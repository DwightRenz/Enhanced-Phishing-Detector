import streamlit as st
import numpy as np
import joblib
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

# -----------------------------------
# Load Models
# -----------------------------------

cnn_model = load_model(
    "cnn_lstm_model.keras"
)

rf_model = joblib.load(
    "enhanced_rf.pkl"
)

tokenizer = joblib.load(
    "tokenizer.pkl"
)

# -----------------------------------
# Build CNN-LSTM Feature Extractor
# -----------------------------------

cnn_model.build(
    input_shape=(None, 200)
)

feature_extractor = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.get_layer(
        "feature_layer"
    ).output
)

# -----------------------------------
# Streamlit Page Config
# -----------------------------------

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------------
# Title
# -----------------------------------

st.title(
    "🛡️ Phishing URL Detection System"
)

st.write(
    "Enhanced Random Forest using CNN-LSTM Feature Extraction"
)

st.markdown("---")

# -----------------------------------
# URL Input
# -----------------------------------

url_input = st.text_input(
    "Enter URL",
    placeholder="https://example.com"
)

# -----------------------------------
# URL Analysis Function
# -----------------------------------

def analyze_url(url):

    suspicious_keywords = [
        "login",
        "verify",
        "secure",
        "account",
        "update",
        "bank",
        "signin",
        "authentication",
        "confirm",
        "security"
    ]

    url_length = len(url)

    dots = url.count('.')

    hyphens = url.count('-')

    has_https = url.startswith("https")

    has_ip = bool(
        re.search(
            r'\d+\.\d+\.\d+\.\d+',
            url
        )
    )

    found_keywords = []

    for word in suspicious_keywords:

        if word in url.lower():

            found_keywords.append(word)

    return {
        "length": url_length,
        "dots": dots,
        "hyphens": hyphens,
        "https": has_https,
        "ip": has_ip,
        "keywords": found_keywords
    }

# -----------------------------------
# Prediction
# -----------------------------------

if st.button("Detect"):

    if url_input.strip() == "":

        st.warning(
            "Please enter a URL"
        )

    else:

        # -----------------------------------
        # Convert URL to Sequence
        # -----------------------------------

        sequence = tokenizer.texts_to_sequences(
            [url_input]
        )

        padded = pad_sequences(
            sequence,
            maxlen=200
        )

        # -----------------------------------
        # CNN-LSTM Feature Extraction
        # -----------------------------------

        features = feature_extractor.predict(
            padded,
            verbose=0
        )

        # -----------------------------------
        # Random Forest Prediction
        # -----------------------------------

        prediction = rf_model.predict(
            features
        )[0]

        probability = rf_model.predict_proba(
            features
        )[0]

        confidence = np.max(
            probability
        )

        legitimate_probability = probability[0] * 100
        phishing_probability = probability[1] * 100

        # -----------------------------------
        # URL Analysis
        # -----------------------------------

        analysis = analyze_url(
            url_input
        )

        st.markdown("---")

        # -----------------------------------
        # Prediction Result
        # -----------------------------------

        if prediction == 1:

            st.error(
                f"⚠️ PHISHING URL DETECTED\n\nConfidence: {confidence:.2%}"
            )

            st.markdown(
                """
                ### Why was this URL flagged?

                The model detected suspicious URL structures and phishing-related patterns commonly used in cyberattacks.
                """
            )

        else:

            st.success(
                f"✅ LEGITIMATE URL\n\nConfidence: {confidence:.2%}"
            )

            st.markdown(
                """
                ### Why was this URL considered safe?

                The URL structure resembles patterns commonly found in legitimate and trusted websites.
                """
            )

        # -----------------------------------
        # Risk Assessment
        # -----------------------------------

        st.markdown("## Risk Assessment")

        if confidence >= 0.90:

            st.error("HIGH CONFIDENCE")

        elif confidence >= 0.70:

            st.warning("MEDIUM CONFIDENCE")

        else:

            st.info("LOW CONFIDENCE")

        # -----------------------------------
        # Probability Statistics
        # -----------------------------------

        st.markdown("## Prediction Statistics")

        st.write(
            f"Legitimate Probability: {legitimate_probability:.2f}%"
        )

        st.write(
            f"Phishing Probability: {phishing_probability:.2f}%"
        )

        # -----------------------------------
        # URL Feature Analysis
        # -----------------------------------

        st.markdown("## URL Feature Analysis")

        table_data = {
            "Feature": [
                "URL Length",
                "Number of Dots",
                "Number of Hyphens",
                "Uses HTTPS",
                "Contains IP Address",
                "Suspicious Keywords"
            ],

            "Result": [
                analysis["length"],
                analysis["dots"],
                analysis["hyphens"],
                analysis["https"],
                analysis["ip"],
                ", ".join(
                    analysis["keywords"]
                ) if analysis["keywords"]
                else "None"
            ]
        }

        st.table(table_data)

        # -----------------------------------
        # Security Interpretation
        # -----------------------------------

        st.markdown("## Security Interpretation")

        if analysis["keywords"]:

            st.write(
                "Suspicious authentication-related keywords were detected in the URL."
            )

        if analysis["hyphens"] > 3:

            st.write(
                "The URL contains many hyphens, which is commonly observed in phishing URLs."
            )

        if not analysis["https"]:

            st.write(
                "The website does not use HTTPS encryption."
            )

        if analysis["ip"]:

            st.write(
                "The URL contains a raw IP address, which is highly suspicious."
            )

        if (
            not analysis["keywords"]
            and analysis["https"]
            and analysis["hyphens"] <= 2
        ):

            st.write(
                "No major suspicious indicators were detected."
            )

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.caption(
    "Final Project: Enhanced Random Forest Classifier Using CNN-LSTM Feature Extraction for Real-Time Phishing URL Detection"
)
