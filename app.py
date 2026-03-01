import streamlit as st
import pandas as pd
import joblib
import os

# Model Path Setup
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")

# Train model automatically if missing
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    from src.train import pipeline
    joblib.dump(pipeline, MODEL_PATH)

# Load model
model = joblib.load(MODEL_PATH)

# UI
st.title("Customer Segmentation Predictor")

income = st.slider("Annual Income (k$)", 0, 150, 50)
score = st.slider("Spending Score (1-100)", 1, 100, 50)

if st.button("Predict Segment"):

    input_df = pd.DataFrame([[income, score]])
    input_df.columns = model.feature_names_in_

    pred = model.predict(input_df)

    st.success(f"Customer belongs to Segment {int(pred[0])}")
