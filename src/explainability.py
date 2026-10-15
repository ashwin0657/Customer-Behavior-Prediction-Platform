import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_COLS = joblib.load("models/feature_cols.pkl")


def plot_feature_importance():
    model = joblib.load("models/churn_model.pkl")
    if not hasattr(model, "feature_importances_"):
        print("Best model has no feature_importances_ (likely Logistic Regression). Use model.coef_ instead.")
        return
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()

    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="#203864")
    plt.title("What drives churn predictions?")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=120)
    print("Saved -> models/feature_importance.png")
    print(importances.sort_values(ascending=False))


def explain_with_shap_LOCAL_ONLY():
    """Run locally after `pip install shap`. Not executed in this sandbox."""
    import shap
    model = joblib.load("models/churn_model.pkl")
    df = pd.read_csv("data/customer_features.csv")
    X = df[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X, show=False)
    plt.savefig("models/shap_summary.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    plot_feature_importance()
