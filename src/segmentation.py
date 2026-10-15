import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = ["Recency", "Frequency", "Monetary"]


def name_cluster(row, medians):
    r, f, m = row["Recency"], row["Frequency"], row["Monetary"]
    if f >= medians["Frequency"] and m >= medians["Monetary"] and r <= medians["Recency"]:
        return "Loyal High-Value"
    if r > medians["Recency"] * 2:
        return "Lapsed / At Risk"
    if f <= 1:
        return "One-Time Buyer"
    return "Occasional Shopper"


def main():
    df = pd.read_csv("data/customer_features.csv")
    X = df[SEG_FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_profile = df.groupby("Cluster")[SEG_FEATURES].mean().round(1)
    print("Cluster profiles (average R/F/M per cluster):")
    print(cluster_profile)

    medians = df[SEG_FEATURES].median()
    cluster_names = df.groupby("Cluster")[SEG_FEATURES].mean().apply(lambda row: name_cluster(row, medians), axis=1)
    df["Segment"] = df["Cluster"].map(cluster_names)

    print("\nSegment sizes:")
    print(df["Segment"].value_counts())

    df.to_csv("data/customer_segments.csv", index=False)
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/segment_scaler.pkl")
    print("\nSaved -> data/customer_segments.csv, models/kmeans_model.pkl")


if __name__ == "__main__":
    main()
