import pandas as pd
import joblib
import os

# Load the trained KMeans model
model_path = os.path.join("..", "models", "kmeans.pkl")
kmeans = joblib.load(model_path)

def predict_cluster(annual_income, spending_score):
    """Predict cluster for a new customer."""
    df = pd.DataFrame({
        'Annual Income (k$)': [annual_income],
        'Spending Score (1-100)': [spending_score]
    })
    cluster = kmeans.predict(df)
    return int(cluster[0])

if __name__ == "__main__":
    print("=== Customer Cluster Prediction ===")
    income = float(input("Enter Annual Income (k$): "))
    score = float(input("Enter Spending Score (1-100): "))
    cluster = predict_cluster(income, score)
    print(f"\nThe customer belongs to cluster: {cluster}")
