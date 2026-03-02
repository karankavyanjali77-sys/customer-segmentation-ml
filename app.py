import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Customer Segmentation Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

# ---------------------------------------------------
# Load or Train Model (Cached)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        from src.train import train_model
        model = train_model()
        joblib.dump(model, MODEL_PATH)
    else:
        model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# ---------------------------------------------------
# Load Data (Cached)
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

data = load_data()

features = ['Annual Income (k$)', 'Spending Score (1-100)']
data["Cluster"] = model.predict(data[features])

# ---------------------------------------------------
# Header Section
# ---------------------------------------------------
st.title("AI-Powered Customer Segmentation Dashboard")
st.markdown("### Intelligent Customer Analytics & Business Insights")

# ---------------------------------------------------
# KPI Section
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(data))
col2.metric("Average Income (k$)", round(data['Annual Income (k$)'].mean(), 2))
col3.metric("Average Spending Score", round(data['Spending Score (1-100)'].mean(), 2))

st.divider()

# ---------------------------------------------------
# User Input Section
# ---------------------------------------------------
st.subheader("Predict Customer Segment")

input_col1, input_col2 = st.columns(2)

with input_col1:
    income = st.slider("Annual Income (k$)", 0, 150, 50)

with input_col2:
    score = st.slider("Spending Score (1-100)", 1, 100, 50)

# ---------------------------------------------------
# Segment Meaning
# ---------------------------------------------------
segment_meaning = {
    0: "High Income – Low Spending (Potential Premium Customers)",
    1: "Low Income – High Spending (Impulse Buyers)",
    2: "Balanced Customers",
    3: "High Income – High Spending (VIP Target Segment)",
    4: "Low Income – Low Spending (Budget Segment)"
}

# ---------------------------------------------------
# Prediction Section
# ---------------------------------------------------
if st.button("Predict Segment"):

    input_df = pd.DataFrame([[income, score]], columns=features)
    pred = model.predict(input_df)[0]

    st.success(f"Customer belongs to Segment {int(pred)}")
    st.info(segment_meaning.get(int(pred), "Segment description unavailable."))

    # ---------------------------------------------------
    # Visualization Section
    # ---------------------------------------------------
    st.subheader("Customer Segmentation Map")

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in sorted(data["Cluster"].unique()):
        cluster_data = data[data["Cluster"] == cluster]

        ax.scatter(
            cluster_data['Annual Income (k$)'],
            cluster_data['Spending Score (1-100)'],
            label=f"Cluster {cluster}",
            alpha=0.6
        )

    # Highlight new customer
    ax.scatter(
        income,
        score,
        s=300,
        marker="X",
        label="New Customer"
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Clusters Distribution")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------------
    # Business Insight Panel
    # ---------------------------------------------------
    st.subheader("Business Insight")

    if pred == 3:
        st.write("👉 This is a VIP customer. Focus on loyalty programs and premium offers.")
    elif pred == 0:
        st.write("👉 High income but low spending. Upselling & targeted promotions recommended.")
    elif pred == 1:
        st.write("👉 Strong engagement. Consider retention strategies.")
    elif pred == 4:
        st.write("👉 Budget-focused segment. Offer discounts & value bundles.")
    else:
        st.write("👉 Balanced customer. Maintain steady engagement.")

    # ---------------------------------------------------
    # Download Prediction Report
    # ---------------------------------------------------
    result_df = pd.DataFrame({
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [score],
        "Predicted Segment": [pred],
        "Segment Meaning": [segment_meaning.get(int(pred))]
    })

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Report",
        data=csv,
        file_name="customer_segment_prediction.csv",
        mime="text/csv"
    )

# ---------------------------------------------------
# Cluster Distribution Overview
# ---------------------------------------------------
st.divider()
st.subheader("Cluster Distribution Overview")

cluster_counts = data["Cluster"].value_counts().sort_index()

fig2, ax2 = plt.subplots()
ax2.bar(cluster_counts.index, cluster_counts.values)
ax2.set_xlabel("Cluster")
ax2.set_ylabel("Number of Customers")
ax2.set_title("Customer Count per Cluster")

st.pyplot(fig2)
plt.close(fig2)
