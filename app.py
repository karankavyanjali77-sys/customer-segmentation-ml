import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    from src.train import pipeline

model = joblib.load(MODEL_PATH)

st.title("Customer Segmentation Predictor")

age = st.slider("Age",18,70)
income = st.slider("Income",10000,100000)

input_df = pd.DataFrame([[age,income]],columns=["Age","Income"])

if st.button("Predict Segment"):

    input_df = pd.DataFrame([[income, score]])

    input_df.columns = model.feature_names_in_

    pred = model.predict(input_df)

    st.success(f"Customer belongs to Segment {int(pred[0])}")
