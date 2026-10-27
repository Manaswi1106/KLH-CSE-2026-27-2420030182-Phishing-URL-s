import joblib
import shap
import numpy as np
import pandas as pd

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/hybrid_model.pkl")
explainer = shap.TreeExplainer(model)

# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "num_digits",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "https",
    "has_ip",
    "url_entropy",
    "suspicious_keywords",
    "html_length",
    "text_length",
    "num_links",
    "external_links",
    "external_ratio",
    "forms",
    "password_fields",
    "login_form",
    "external_forms",
    "external_form_ratio",
    "title_length",
    "iframe_count"
]

for i in range(768):
    FEATURE_NAMES.append(f"bert_{i}")

assert len(FEATURE_NAMES) == 791, (
    f"Expected 791 feature names, got {len(FEATURE_NAMES)}"
)

# ============================================================
# SHAP FUNCTION
# ============================================================

def explain_prediction(fused_vector):

    fused_vector = np.asarray(fused_vector, dtype=np.float32)

    if fused_vector.ndim == 1:
        fused_vector = fused_vector.reshape(1, -1)

    if fused_vector.shape[1] != 791:
        raise ValueError(
            f"Expected 791 features, got {fused_vector.shape[1]}"
        )

    values = explainer.shap_values(fused_vector)

    # Handle different SHAP versions
    if isinstance(values, list):
        values = values[1]

    # Newer SHAP versions may return shape:
    # (samples, features, classes)
    elif isinstance(values, np.ndarray) and values.ndim == 3:
        values = values[:, :, 1]

    scores = values[0]

    pairs = list(zip(FEATURE_NAMES, scores))

    pairs = sorted(
        pairs,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return [
        {
            "feature": feature,
            "impact": round(float(score), 6)
        }
        for feature, score in pairs[:10]
    ]


# ============================================================
# GLOBAL SHAP ANALYSIS
# ============================================================
#
# This section actually RUNS when:
#     python shap_explainer.py
#
# It calculates SHAP importance over the Hybrid test data.
# ============================================================
if __name__ == "__main__":
    print("\n========== PHISHXPERT SHAP ANALYSIS ==========\n")

    # Load hybrid dataset
    df = pd.read_csv("data/hybrid_dataset.csv")

    X = df.drop("label", axis=1)

    print(f"Dataset shape : {X.shape}")
    print(f"Feature count : {X.shape[1]}")

    # Use a manageable sample for SHAP
    # The model was trained on the same 791-dimensional representation.
    SAMPLE_SIZE = min(500, len(X))

    X_sample = X.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )

    print(f"SHAP samples  : {len(X_sample)}")
    print("\nCalculating SHAP values...")

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Global mean absolute SHAP
    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_abs_shap": mean_abs_shap
    })

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    importance.to_csv(
        "shap_feature_importance.csv",
        index=False
    )

    top10 = importance.head(10)

    print("\n========== TOP 10 SHAP FEATURES ==========\n")

    for i, row in top10.iterrows():
        print(
            f"{i+1:2d}. "
            f"{row['feature']:30s} "
            f"{row['mean_abs_shap']:.6f}"
        )

    # Top engineered features only
    engineered = importance[
        ~importance["feature"].str.startswith("bert_")
    ].head(10)

    print("\n========== TOP ENGINEERED FEATURES ==========\n")

    for i, row in engineered.iterrows():
        print(
            f"{i+1:2d}. "
            f"{row['feature']:30s} "
            f"{row['mean_abs_shap']:.6f}"
        )

    print("\nSaved:")
    print("  shap_feature_importance.csv")

    print("\n========== SHAP ANALYSIS COMPLETE ==========\n")
