import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

features = ['Annual Income (k$)', 'Spending Score (1-100)']

def train_model():
    data = pd.read_csv(DATA_PATH)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=5, random_state=42))
    ])

    pipeline.fit(data[features])

    return pipeline
