import os
import joblib

MODEL_PATH = "models/kmeans.pkl"

def get_model():
    if not os.path.exists(MODEL_PATH):
        import src.train
    return joblib.load(MODEL_PATH)
