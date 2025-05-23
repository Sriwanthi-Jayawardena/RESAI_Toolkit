import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap
import numpy as np
import xgboost as xgb

def calculate_dynamic_threshold(bias_scores):
    mean_bias = np.mean(bias_scores)
    std_bias = np.std(bias_scores)
    male_threshold = mean_bias + 0.5 * std_bias
    female_threshold = mean_bias - 0.5 * std_bias
    return male_threshold, female_threshold

def run_image_bias_pipeline():
    df = pd.read_csv("XAI_validation_data_new.csv")

    # Encode Gender
    df['Gender'] = df['Gender'].astype(str).str.lower().map({'male': 1, 'female': 0})
    df = df.dropna(subset=['Gender'])

    # SHAP + PCA weights
    features = ["Relative_Size", "3D_Distance", "Normalized_Depth"]
    target = "Gender"
    X = df[features]
    y = df[target]

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_balanced, y_balanced)

    explainer = shap.Explainer(model, X_balanced)
    shap_values = explainer(X_balanced)
    shap_weights = np.abs(shap_values.values).mean(axis=0)

    pca = PCA(n_components=1)
    pca.fit(X_balanced)
    pca_weights = np.abs(pca.components_[0])

    combined_weights = 0.6 * shap_weights + 0.4 * pca_weights
    combined_weights /= combined_weights.sum()

    # OIS Calculation
    df["OIS"] = np.dot(df[features], combined_weights)

    # Aggregate
    grouped = df.groupby(["Object_Class", "Gender"]).agg(
        Mean_OIS=("OIS", "mean"),
        Mean_SSB=("Scene_Similarity_Bias", "mean")
    ).reset_index()

    pivot = grouped.pivot(index="Object_Class", columns="Gender", values=["Mean_OIS", "Mean_SSB"]).reset_index()
    pivot.columns = ["Object_Class", "OIS_Female", "OIS_Male", "SSB_Female", "SSB_Male"]
    pivot.fillna(0, inplace=True)

    pivot["Label"] = np.where(pivot["OIS_Male"] >= pivot["OIS_Female"], 1, 0)

    X_ridge = pivot[["OIS_Male", "SSB_Male"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ridge)

    ridge = Ridge()
    ridge.fit(X_scaled, pivot["Label"])
    alpha, beta = np.abs(ridge.coef_) / np.sum(np.abs(ridge.coef_))

    pivot["Bias_Score"] = ((alpha * (pivot["OIS_Male"] - pivot["OIS_Female"])) +
                           (beta * (pivot["SSB_Male"] - pivot["SSB_Female"]))) / (
                           (alpha * (pivot["OIS_Male"] + pivot["OIS_Female"])) +
                           (beta * (pivot["SSB_Male"] + pivot["SSB_Female"])) + 1e-6)

    male_thresh, female_thresh = calculate_dynamic_threshold(pivot["Bias_Score"])
    pivot["Bias_Category"] = pivot["Bias_Score"].apply(
        lambda x: "Male-Biased" if x > male_thresh else ("Female-Biased" if x < female_thresh else "Neutral")
    )

    return pivot[["Object_Class", "Bias_Score", "Bias_Category"]]

# def run_image_bias_pipeline_from_csv(csv_path):
#     df = pd.read_csv(csv_path)

#     df['Gender'] = df['Gender'].astype(str).str.lower().map({'male': 1, 'female': 0})
#     df = df.dropna(subset=['Gender'])

#     features = ["Relative_Size", "3D_Distance", "Normalized_Depth"]
#     X = df[features]
#     y = df["Gender"]

#     smote = SMOTE(random_state=42)
#     X_balanced, y_balanced = smote.fit_resample(X, y)

#     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#     model.fit(X_balanced, y_balanced)

#     explainer = shap.Explainer(model, X_balanced)
#     shap_values = explainer(X_balanced)
#     shap_weights = np.abs(shap_values.values).mean(axis=0)

#     pca = PCA(n_components=1)
#     pca.fit(X_balanced)
#     pca_weights = np.abs(pca.components_[0])

#     combined_weights = 0.6 * shap_weights + 0.4 * pca_weights
#     combined_weights /= combined_weights.sum()

#     df["OIS"] = np.dot(df[features], combined_weights)

#     grouped = df.groupby(["Object_Class", "Gender"]).agg(
#         Mean_OIS=("OIS", "mean"),
#         Mean_SSB=("Scene_Similarity_Bias", "mean")
#     ).reset_index()

#     pivot = grouped.pivot(index="Object_Class", columns="Gender", values=["Mean_OIS", "Mean_SSB"]).reset_index()
#     pivot.columns = ["Object_Class", "OIS_Female", "OIS_Male", "SSB_Female", "SSB_Male"]
#     pivot.fillna(0, inplace=True)

#     pivot["Label"] = np.where(pivot["OIS_Male"] >= pivot["OIS_Female"], 1, 0)

#     X_ridge = pivot[["OIS_Male", "SSB_Male"]]
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_ridge)

#     ridge = Ridge()
#     ridge.fit(X_scaled, pivot["Label"])
#     alpha, beta = np.abs(ridge.coef_) / np.sum(np.abs(ridge.coef_))

#     pivot["Bias_Score"] = ((alpha * (pivot["OIS_Male"] - pivot["OIS_Female"])) +
#                            (beta * (pivot["SSB_Male"] - pivot["SSB_Female"]))) / (
#                            (alpha * (pivot["OIS_Male"] + pivot["OIS_Female"])) +
#                            (beta * (pivot["SSB_Male"] + pivot["SSB_Female"])) + 1e-6)

#     male_thresh, female_thresh = calculate_dynamic_threshold(pivot["Bias_Score"])
#     pivot["Bias_Category"] = pivot["Bias_Score"].apply(
#         lambda x: "Male-Biased" if x > male_thresh else ("Female-Biased" if x < female_thresh else "Neutral")
#     )

#     return pivot.sort_values(by="Bias_Score", ascending=False)


def run_image_bias_pipeline_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # ✅ Column check
    required_cols = ["Object_Class", "Gender", "Relative_Size", "3D_Distance", "Normalized_Depth", "Scene_Similarity_Bias"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing columns in CSV: {', '.join(missing_cols)}")

    # ✅ Gender format handling
    if df['Gender'].dtype == object or df['Gender'].astype(str).str.lower().isin(['male', 'female']).any():
        df['Gender'] = df['Gender'].astype(str).str.lower().map({'male': 1, 'female': 0})
    else:
        df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')

    df = df[df['Gender'].isin([0, 1])]
    if df.empty:
        raise ValueError("❌ No valid gender data found (expected 'male', 'female', or encoded 0/1).")

    # ✅ SMOTE-ready features
    features = ["Relative_Size", "3D_Distance", "Normalized_Depth"]
    X = df[features]
    y = df["Gender"]

    if X.empty or y.empty:
        raise ValueError("❌ Cannot train model: Features or labels are empty after filtering.")

    # ✅ SMOTE balancing
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # ✅ Train model + SHAP
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_balanced, y_balanced)

    explainer = shap.Explainer(model, X_balanced)
    shap_values = explainer(X_balanced)
    shap_weights = np.abs(shap_values.values).mean(axis=0)

    # ✅ PCA weighting
    pca = PCA(n_components=1)
    pca.fit(X_balanced)
    pca_weights = np.abs(pca.components_[0])

    combined_weights = 0.6 * shap_weights + 0.4 * pca_weights
    combined_weights /= combined_weights.sum()

    # ✅ OIS computation
    df["OIS"] = np.dot(df[features], combined_weights)

    grouped = df.groupby(["Object_Class", "Gender"]).agg(
        Mean_OIS=("OIS", "mean"),
        Mean_SSB=("Scene_Similarity_Bias", "mean")
    ).reset_index()

    pivot = grouped.pivot(index="Object_Class", columns="Gender", values=["Mean_OIS", "Mean_SSB"]).reset_index()
    pivot.columns = ["Object_Class", "OIS_Female", "OIS_Male", "SSB_Female", "SSB_Male"]
    pivot.fillna(0, inplace=True)

    pivot["Label"] = np.where(pivot["OIS_Male"] >= pivot["OIS_Female"], 1, 0)

    # ✅ Ridge Regression weights
    X_ridge = pivot[["OIS_Male", "SSB_Male"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ridge)

    ridge = Ridge()
    ridge.fit(X_scaled, pivot["Label"])
    alpha, beta = np.abs(ridge.coef_) / np.sum(np.abs(ridge.coef_))

    # ✅ Bias Score
    pivot["Bias_Score"] = ((alpha * (pivot["OIS_Male"] - pivot["OIS_Female"])) +
                           (beta * (pivot["SSB_Male"] - pivot["SSB_Female"]))) / (
                           (alpha * (pivot["OIS_Male"] + pivot["OIS_Female"])) +
                           (beta * (pivot["SSB_Male"] + pivot["SSB_Female"])) + 1e-6)

    # ✅ Category assignment
    male_thresh, female_thresh = calculate_dynamic_threshold(pivot["Bias_Score"])
    pivot["Bias_Category"] = pivot["Bias_Score"].apply(
        lambda x: "Male-Biased" if x > male_thresh else ("Female-Biased" if x < female_thresh else "Neutral")
    )

    return pivot.sort_values(by="Bias_Score", ascending=False)
