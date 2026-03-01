import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

# ------------------------
# Load model
# ------------------------
model = joblib.load(MODEL_PATH)

# ------------------------
# Load dataset for visualization
# ------------------------
data = pd.read_csv(DATA_PATH)
features = ['Annual Income (k$)', 'Spending Score (1-100)']

# Predict clusters for dataset
data["Cluster"] = model.predict(data[features])

# ------------------------
# UI
# ------------------------
st.title("Customer Segmentation Predictor")

income = st.slider("Annual Income (k$)", 0, 150, 50)
score = st.slider("Spending Score (1-100)", 1, 100, 50)

if st.button("Predict Segment"):

    input_df = pd.DataFrame([[income, score]])
    input_df.columns = model.feature_names_in_

    pred = model.predict(input_df)[0]

    st.success(f"Customer belongs to Segment {int(pred)}")

    # ------------------------
    # Plot clusters
    # ------------------------
    fig, ax = plt.subplots()

    for cluster in data["Cluster"].unique():
        cluster_data = data[data["Cluster"] == cluster]
        ax.scatter(
            cluster_data['Annual Income (k$)'],
            cluster_data['Spending Score (1-100)'],
            label=f"Cluster {cluster}",
            alpha=0.6
        )

    # Highlight new customer
    ax.scatter(income, score, s=200, marker="X", label="New Customer")

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend()

    st.pyplot(fig)
