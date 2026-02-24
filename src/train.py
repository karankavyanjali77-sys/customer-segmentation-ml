import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("data/data.csv")

# Select features
features = ['Annual Income (k$)', 'Spending Score (1-100)']

# Creating pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=5, random_state=42))
])

# Train model
pipeline.fit(data[features])

# Save model
joblib.dump(pipeline, "models/kmeans.pkl")

print("Model trained and saved!")
