import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

FEATURE_COLS = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue",
    "TenureDays", "UniqueProducts", "ReturnRate", "PurchaseFrequencyRate",
]


def load_data(path="data/customer_features.csv"):
    df = pd.read_csv(path)
    return df, df[FEATURE_COLS], df["Churned"]


def evaluate(name, model, X_test, y_test, scaled=False, scaler=None):
    X_eval = scaler.transform(X_test) if scaled else X_test
    preds = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]
    print(f"\n--- {name} ---")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1:        {f1_score(y_test, preds):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, proba):.3f}")
    return {"model": name, "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds), "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba)}


def main():
    df, X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = []
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_s, y_train)
    results.append(evaluate("Logistic Regression", logreg, X_test, y_test, scaled=True, scaler=scaler))

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    try:
        from xgboost import XGBClassifier
        gb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42)
        model_name = "XGBoost"
    except ImportError:
        gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, random_state=42)
        model_name = "Gradient Boosting (sklearn fallback - install xgboost for the real thing)"
    gb.fit(X_train, y_train)
    results.append(evaluate(model_name, gb, X_test, y_test))

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_comparison.csv", index=False)
    print("\n=== Model comparison saved to models/model_comparison.csv ===")
    print(results_df)

    best_idx = results_df["roc_auc"].idxmax()
    best_name = results_df.loc[best_idx, "model"]
    best_model = {"Logistic Regression": logreg, "Random Forest": rf}.get(best_name, gb)
    joblib.dump(best_model, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(FEATURE_COLS, "models/feature_cols.pkl")
    joblib.dump(best_name, "models/best_model_name.pkl")
    print(f"\nBest model: {best_name} -> saved to models/churn_model.pkl")


if __name__ == "__main__":
    main()
