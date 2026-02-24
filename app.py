import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("models/kmeans.pkl","rb"))

st.title("Customer Segmentation Predictor")

age = st.slider("Age",18,70)
income = st.slider("Income",10000,100000)

input_df = pd.DataFrame([[age,income]],columns=["Age","Income"])

if st.button("Predict Segment"):
    pred = model.predict(input_df)
    st.success(f"Customer belongs to Segment {pred[0]}")
