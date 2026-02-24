import pandas as pd
import joblib

model = joblib.load("../models/kmeans.pkl")

def predict_cluster(income, spending_score):
    df = pd.DataFrame({'Annual Income (k$)': [income],
                       'Spending Score (1-100)': [spending_score]})
    cluster = model.predict(df)
    return int(cluster[0])

# Example usage
if __name__ == "__main__":
    cluster = predict_cluster(50, 60)
    print(f"The customer belongs to cluster {cluster}")
