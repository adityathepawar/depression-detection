import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import joblib
import tempfile
from features.extract_features import extract_features

model = joblib.load("../models/depression_model.pkl")

st.title("Depression Detection from Voice")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())

    features = extract_features(temp.name).reshape(1,-1)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Depression Detected")
    else:
        st.success("No Depression Detected")