🚀 AI-Powered Customer Segmentation Dashboard

An end-to-end Machine Learning project that builds and deploys a Customer Segmentation System using K-Means clustering, integrated with a professional Streamlit analytics dashboard.

This project transforms raw customer data into actionable business intelligence insights.

📌 Project Overview

This application:

Segments customers using K-Means Clustering

Provides real-time segment prediction

Generates business insights for decision-making

Displays KPI analytics

Visualizes cluster distributions

Allows downloading prediction reports

It demonstrates the complete ML lifecycle:
Data → Model Training → Inference → Visualization → Deployment

🧠 Problem Statement

Businesses need to understand customer behavior to:

Improve marketing targeting

Increase retention

Identify VIP customers

Optimize promotional strategies

This project solves that using unsupervised machine learning.

🛠️ Tech Stack

Python

Pandas

Scikit-learn

Matplotlib

Joblib

Streamlit

📊 Features
✅ 1. Real-Time Customer Segment Prediction

Input:

Annual Income (k$)

Spending Score (1–100)

Output:

Predicted Cluster

Segment Meaning

Business Recommendation

✅ 2. AI-Powered Analytics Dashboard

Includes:

Total Customers KPI

Average Income KPI

Average Spending Score KPI

Cluster Distribution Visualization

Customer Segmentation Scatter Plot

✅ 3. Business Insight Engine

Each predicted segment generates strategic recommendations such as:

VIP loyalty programs

Upselling opportunities

Retention strategies

Discount targeting

✅ 4. Downloadable Prediction Report

Users can download a structured CSV report containing:

Customer Input

Predicted Segment

Segment Meaning

🏗️ Project Architecture
customer-segmentation-ml/
│
├── app.py                 # Streamlit AI dashboard
├── data/
│   └── data.csv           # Customer dataset
├── models/
│   └── kmeans.pkl         # Trained clustering model (auto-generated)
├── src/
│   └── train.py           # Model training logic
├── requirements.txt       # Python dependencies
├── .devcontainer.json     # Development container configuration
└── README.md              # Project documentation
⚙️ How It Works

Data is loaded and preprocessed

A Scikit-learn Pipeline:

StandardScaler

KMeans (5 clusters)

Model is saved using Joblib

Streamlit loads the model

User inputs are predicted in real-time

Dashboard visualizes cluster intelligence

📈 Machine Learning Details

Algorithm: K-Means Clustering

Features Used:

Annual Income (k$)

Spending Score (1–100)

Clusters: 5

Random State: 42

Scaling: StandardScaler

💼 Business Value

This dashboard can be used by:

Marketing teams

Retail analytics teams

CRM teams

Business intelligence analysts

It converts unsupervised ML output into actionable strategic insights.

🧪 How to Run Locally
git clone <your-repo-url>
cd customer-segmentation-ml
pip install -r requirements.txt
streamlit run app.py
🎯 Key Learning Outcomes

Built modular ML pipeline

Implemented model persistence

Designed business-ready dashboard

Integrated caching for performance

Converted clustering into decision-support system

Deployed ML application to cloud

🏆 Resume Highlight

Developed and deployed an AI-powered Customer Segmentation Dashboard integrating clustering analytics, KPI intelligence, business insight generation, and real-time prediction using Scikit-learn and Streamlit.

🔮 Future Improvements (Optional Expansion)

Dynamic cluster tuning (Elbow method UI)

Multi-model comparison

User authentication

Database integration

Advanced interactive plotting (Plotly)

🚀 Author

Kavyanjali Karan
B.Tech Computer Science Engineering
AI & Machine Learning Enthusiast
