import joblib
from features.extract_features import extract_features

model = joblib.load("models/depression_model.pkl")

audio = "data/300_P/300_AUDIO.wav"

features = extract_features(audio).reshape(1,-1)

prediction = model.predict(features)

if prediction[0] == 1:
    print("Depression Detected")
else:
    print("No Depression Detected")