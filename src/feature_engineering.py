import pandas as pd
import numpy as np

CHURN_WINDOW_DAYS = 90


def build_customer_features(df: pd.DataFrame, cutoff_date=None):
    df = df.copy()
    if cutoff_date is None:
        max_date = df["InvoiceDate"].max()
        cutoff_date = max_date - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    cutoff_date = pd.Timestamp(cutoff_date)

    history = df[df["InvoiceDate"] <= cutoff_date]
    future = df[(df["InvoiceDate"] > cutoff_date) &
                (df["InvoiceDate"] <= cutoff_date + pd.Timedelta(days=CHURN_WINDOW_DAYS))]

    grouped = history.groupby("CustomerID")
    features = grouped.agg(
        Recency=("InvoiceDate", lambda x: (cutoff_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        AvgOrderValue=("TotalPrice", "mean"),
        TenureDays=("InvoiceDate", lambda x: (cutoff_date - x.min()).days),
        UniqueProducts=("StockCode", "nunique"),
        ReturnRate=("IsReturn", "mean"),
    ).reset_index()

    features["TenureDays"] = features["TenureDays"].clip(lower=1)
    features["PurchaseFrequencyRate"] = features["Frequency"] / features["TenureDays"] * 30

    returning_customers = set(future["CustomerID"].unique())
    features["Churned"] = (~features["CustomerID"].isin(returning_customers)).astype(int)

    return features, cutoff_date


if __name__ == "__main__":
    df = pd.read_csv("data/clean_transactions.csv", parse_dates=["InvoiceDate"])
    features, cutoff = build_customer_features(df)
    features.to_csv("data/customer_features.csv", index=False)
    print(f"Cutoff date used: {cutoff.date()}")
    print(f"Customers with history before cutoff: {len(features)}")
    print(f"Churn rate in dataset: {features['Churned'].mean():.2%}")
    print(features.head())
