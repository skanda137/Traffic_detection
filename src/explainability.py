import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from model_comparison import run_temporal_split

def explain_model():
    # 1. Load validated data and temporal holdout
    df = pd.read_csv("synthetic_traffic_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    X_tr, y_tr, X_va, y_va = run_temporal_split(df)
    
    # 2. Load trained model artifact
    model = joblib.load("best_congestion_model.pkl")
    
    # 3. Initialize SHAP TreeExplainer
    print("Computing SHAP values on validation set...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_va)
    
    # 4. Generate & Save Global Feature Importance Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_va, show=False)
    plt.title("SHAP Feature Importance: Exogenous Impact on Congestion Ratio", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300)
    plt.close()
    print("Saved global attribution plot to shap_summary_plot.png")
    
    # 5. Extract Mean Absolute SHAP per Feature for Viva Evidence
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_va.columns,
        "mean_absolute_shap": mean_abs_shap
    }).sort_values(by="mean_absolute_shap", ascending=False)
    
    print("\nFeature Impact Ranking (Mean |SHAP|):")
    print(importance_df.to_string(index=False))
    importance_df.to_csv("shap_feature_importance.csv", index=False)

if __name__ == "__main__":
    explain_model()