import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Customer Behavior Prediction Platform", layout="wide")
st.title("📊 Customer Behavior Prediction Platform")
st.caption("Churn risk, lifetime value signals, and customer segments — built end-to-end from raw transactions.")


@st.cache_data
def load_data():
    features = pd.read_csv(os.path.join(BASE, "data", "customer_features.csv"))
    segments = pd.read_csv(os.path.join(BASE, "data", "customer_segments.csv"))
    return features, segments


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE, "models", "churn_model.pkl"))
    feature_cols = joblib.load(os.path.join(BASE, "models", "feature_cols.pkl"))
    best_name = joblib.load(os.path.join(BASE, "models", "best_model_name.pkl"))
    return model, feature_cols, best_name


features, segments = load_data()
model, feature_cols, best_name = load_model()
features["ChurnProbability"] = model.predict_proba(features[feature_cols])[:, 1]

tab1, tab2, tab3 = st.tabs(["Overview", "Customer Lookup", "Segments"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(features):,}")
    col2.metric("Predicted Churn Rate", f"{(features['ChurnProbability'] >= 0.5).mean():.1%}")
    col3.metric("Avg Customer Value", f"₹{features['Monetary'].mean():,.0f}")
    col4.metric("Model Used", best_name)

    st.subheader("Churn Risk Distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(features["ChurnProbability"], bins=30, color="#203864")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("Top 10 Highest-Risk Customers")
    top_risk = features.sort_values("ChurnProbability", ascending=False).head(10)
    st.dataframe(top_risk[["CustomerID", "ChurnProbability", "Recency", "Frequency", "Monetary"]])

with tab2:
    st.subheader("Look up a customer")
    cust_id = st.selectbox("Select Customer ID", features["CustomerID"].sort_values().tolist())
    row = features[features["CustomerID"] == cust_id].iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Churn Probability", f"{row['ChurnProbability']:.1%}")
        st.metric("Recency (days since last order)", int(row["Recency"]))
        st.metric("Total Orders", int(row["Frequency"]))
    with c2:
        st.metric("Lifetime Spend", f"₹{row['Monetary']:,.0f}")
        st.metric("Avg Order Value", f"₹{row['AvgOrderValue']:,.0f}")
        st.metric("Unique Products Bought", int(row["UniqueProducts"]))

with tab3:
    st.subheader("Customer Segments")
    seg_counts = segments["Segment"].value_counts()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(seg_counts.index, seg_counts.values, color="#2E75B6")
    plt.xticks(rotation=20)
    st.pyplot(fig2)

    st.dataframe(segments.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1))
