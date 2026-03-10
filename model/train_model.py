import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from features.extract_features import extract_features

audio_path = "data/300_P/300_AUDIO.wav"

X = []
y = []

features = extract_features(audio_path)

X.append(features)

# dummy label for demo
y.append(1)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "models/depression_model.pkl")

print("Model trained and saved.")