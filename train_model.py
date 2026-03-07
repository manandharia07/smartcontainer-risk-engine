"""
SmartContainer Risk Engine - train_model.py
===========================================
Full pipeline: Feature Engineering -to- Anomaly Detection -to- SMOTE + LightGBM -to-
SHAP Explainability -to- Risk Score -to- Output CSV + Saved Model.

Run:  python train_model.py
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, recall_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# --------------------------------------------─
# 0. CONFIG
# --------------------------------------------─
DATA_PATH   = "Historical Data.csv"
MODEL_PATH  = "risk_model.pkl"
ENC_PATH    = "label_encoders.pkl"
OUT_PATH    = "container_risk_predictions.csv"

# Risk score thresholds (0-100 scale)
CRITICAL_THRESH  = 55   # Risk_Score >= CRITICAL_THRESH  -to- Critical
LOW_RISK_THRESH  = 22   # Risk_Score >= LOW_RISK_THRESH  -to- Low Risk
                         # Risk_Score <  LOW_RISK_THRESH  -to- Clear

LABEL_MAP   = {"Clear": 0, "Low Risk": 1, "Critical": 2}
INV_LABEL   = {v: k for k, v in LABEL_MAP.items()}

RANDOM_STATE = 42
N_FOLDS      = 5

# --------------------------------------------─
# 1. LOAD DATA
# --------------------------------------------─
print("=" * 60)
print("SmartContainer Risk Engine - Training Pipeline")
print("=" * 60)

print(f"\n[1/7] Loading data from '{DATA_PATH}' ...")
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}")

# Normalise column names
df.columns = [c.strip() for c in df.columns]
df.rename(columns={
    "Declaration_Date (YYYY-MM-DD)":         "Declaration_Date",
    "Trade_Regime (Import / Export / Transit)": "Trade_Regime",
}, inplace=True)

# --------------------------------------------─
# 2. FEATURE ENGINEERING
# --------------------------------------------─
print("\n[2/7] Feature engineering ...")

# --- 2a. Date / Time features
df["Declaration_Date"] = pd.to_datetime(df["Declaration_Date"], errors="coerce")
df["hour_of_day"]  = pd.to_datetime(df["Declaration_Time"], format="%H:%M:%S", errors="coerce").dt.hour.fillna(12)
df["day_of_week"]  = df["Declaration_Date"].dt.dayofweek.fillna(0)
df["month"]        = df["Declaration_Date"].dt.month.fillna(1)
df["year"]         = df["Declaration_Date"].dt.year.fillna(2020)

# --- 2b. Weight features
df["weight_diff_abs"] = df["Measured_Weight"] - df["Declared_Weight"]
df["weight_diff_pct"] = (
    df["weight_diff_abs"] / (df["Declared_Weight"].abs() + 0.001) * 100
)
df["weight_mismatch_flag"]  = (df["weight_diff_pct"].abs() > 5).astype(int)
df["weight_mismatch_severe"]= (df["weight_diff_pct"].abs() > 15).astype(int)
df["weight_ratio"]          = df["Measured_Weight"] / (df["Declared_Weight"] + 0.001)

# --- 2c. Value features
df["value_per_kg"]          = df["Declared_Value"] / (df["Declared_Weight"] + 0.001)
df["log_declared_value"]    = np.log1p(df["Declared_Value"].clip(lower=0))
df["log_declared_weight"]   = np.log1p(df["Declared_Weight"].clip(lower=0))
df["log_measured_weight"]   = np.log1p(df["Measured_Weight"].clip(lower=0))
df["log_dwell_time"]        = np.log1p(df["Dwell_Time_Hours"].clip(lower=0))
df["log_value_per_kg"]      = np.log1p(df["value_per_kg"].clip(lower=0))
df["zero_value_flag"]       = (df["Declared_Value"] == 0).astype(int)

# --- 2d. HS-Code chapter (first 2 digits -to- high-risk chapter)
df["hs_chapter"] = (df["HS_Code"] // 10000).astype(int)

# --- 2e. Trade Regime flag
df["is_transit"] = (df["Trade_Regime"].str.strip() == "Transit").astype(int)

# --- 2f. Historical entity risk rates (leak-safe: computed on full dataset
#         for training; for predict.py they are precomputed and saved)
target_num = df["Clearance_Status"].map(LABEL_MAP)

def entity_risk_rate(df_in, col, label_series, critical_val=2, smooth=10):
    """Smoothed critical proportion per entity."""
    group = pd.DataFrame({"entity": df_in[col], "y": label_series})
    stats = group.groupby("entity")["y"].agg(
        count="count",
        pos=lambda x: (x == critical_val).sum()
    ).reset_index()
    global_rate = (label_series == critical_val).mean()
    stats["rate"] = (stats["pos"] + smooth * global_rate) / (stats["count"] + smooth)
    return df_in[col].map(stats.set_index("entity")["rate"]).fillna(global_rate)

print("      Computing entity risk rates ...")
df["importer_risk_rate"]     = entity_risk_rate(df, "Importer_ID",        target_num)
df["exporter_risk_rate"]     = entity_risk_rate(df, "Exporter_ID",        target_num)
df["country_risk_rate"]      = entity_risk_rate(df, "Origin_Country",     target_num)
df["port_risk_rate"]         = entity_risk_rate(df, "Destination_Port",   target_num)
df["shipping_line_risk_rate"]= entity_risk_rate(df, "Shipping_Line",      target_num)
df["dest_country_risk_rate"] = entity_risk_rate(df, "Destination_Country",target_num)

# Route = Origin -to- Destination
df["route"] = df["Origin_Country"].astype(str) + "_" + df["Destination_Country"].astype(str)
df["route_risk_rate"] = entity_risk_rate(df, "route", target_num)

# Combined risk score (weighted sum of entity rates)
df["entity_risk_combined"] = (
    0.30 * df["importer_risk_rate"] +
    0.20 * df["exporter_risk_rate"] +
    0.15 * df["country_risk_rate"] +
    0.10 * df["port_risk_rate"] +
    0.10 * df["shipping_line_risk_rate"] +
    0.10 * df["dest_country_risk_rate"] +
    0.05 * df["route_risk_rate"]
)

# --- 2g. Label encode high-cardinality categoricals
print("      Label-encoding categoricals ...")
cat_cols = ["Origin_Country", "Destination_Country", "Destination_Port",
            "Shipping_Line", "Importer_ID", "Exporter_ID", "Trade_Regime"]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# --- 2h. Save risk-rate lookup tables for predict.py
risk_rate_tables = {}
for col, rate_col in [
    ("Importer_ID",         "importer_risk_rate"),
    ("Exporter_ID",         "exporter_risk_rate"),
    ("Origin_Country",      "country_risk_rate"),
    ("Destination_Port",    "port_risk_rate"),
    ("Shipping_Line",       "shipping_line_risk_rate"),
    ("Destination_Country", "dest_country_risk_rate"),
    ("route",               "route_risk_rate"),
]:
    risk_rate_tables[rate_col] = df.groupby(col)[rate_col].first().to_dict()

joblib.dump(risk_rate_tables, "risk_rate_tables.pkl")
print("      Saved risk_rate_tables.pkl")

# --------------------------------------------─
# 3. ISOLATION FOREST ANOMALY DETECTION
# --------------------------------------------─
print("\n[3/7] Isolation Forest anomaly detection ...")

iso_features = [
    "log_declared_value", "log_declared_weight", "log_measured_weight",
    "weight_diff_pct", "weight_ratio", "log_dwell_time", "log_value_per_kg",
    "hour_of_day", "hs_chapter", "importer_risk_rate", "exporter_risk_rate",
    "country_risk_rate", "is_transit"
]

iso = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    max_samples="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso.fit(df[iso_features])
df["anomaly_score"]   = -iso.score_samples(df[iso_features])   # higher = more anomalous
df["is_anomaly"]      = (iso.predict(df[iso_features]) == -1).astype(int)

print(f"      Anomalies detected: {df['is_anomaly'].sum()} "
      f"({df['is_anomaly'].mean()*100:.1f}%)")

joblib.dump(iso, "isolation_forest.pkl")

# --------------------------------------------─
# 4. DEFINE FEATURE SET & TARGET
# --------------------------------------------─
FEATURES = [
    # Weight / physical
    "weight_diff_abs", "weight_diff_pct", "weight_mismatch_flag",
    "weight_mismatch_severe", "weight_ratio",
    # Value
    "log_declared_value", "log_declared_weight", "log_measured_weight",
    "log_dwell_time", "value_per_kg", "log_value_per_kg", "zero_value_flag",
    # Time
    "hour_of_day", "day_of_week", "month", "year",
    # HS Code
    "hs_chapter",
    # Trade regime
    "is_transit",
    # Entity risk rates
    "importer_risk_rate", "exporter_risk_rate", "country_risk_rate",
    "port_risk_rate", "shipping_line_risk_rate", "dest_country_risk_rate",
    "route_risk_rate", "entity_risk_combined",
    # Anomaly
    "anomaly_score", "is_anomaly",
    # Encoded categoricals
    "Origin_Country_enc", "Destination_Country_enc", "Destination_Port_enc",
    "Shipping_Line_enc", "Importer_ID_enc", "Exporter_ID_enc",
]

X = df[FEATURES].copy()
y = target_num.copy()

print(f"\n      Feature set: {len(FEATURES)} features")
print(f"      Target distribution:\n{y.value_counts().sort_index()}")

# --------------------------------------------─
# 5. CROSS-VALIDATION
# --------------------------------------------─
print("\n[4/7] Stratified 5-Fold Cross-Validation ...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

lgb_params = dict(
    objective        = "multiclass",
    num_class        = 3,
    n_estimators     = 1200,
    learning_rate    = 0.04,
    num_leaves       = 63,
    max_depth        = 8,
    min_child_samples= 10,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    class_weight     = "balanced",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    verbosity        = -1,
)

# SMOTE settings - oversample minority classes to 50% of majority
smote = SMOTE(
    sampling_strategy  = "not majority",
    k_neighbors        = 5,
    random_state       = RANDOM_STATE,
)

fold_macro_f1    = []
fold_critical_f1 = []
fold_critical_rc = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # SMOTE oversampling on training fold
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        X_tr_res, y_tr_res,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(60, verbose=False),
                   lgb.log_evaluation(-1)]
    )
    y_pred = clf.predict(X_val)

    macro_f1    = f1_score(y_val, y_pred, average="macro")
    critical_f1 = f1_score(y_val, y_pred, average=None, labels=[2])[0]
    critical_rc = recall_score(y_val, y_pred, average=None, labels=[2])[0]

    fold_macro_f1.append(macro_f1)
    fold_critical_f1.append(critical_f1)
    fold_critical_rc.append(critical_rc)

    print(f"      Fold {fold}: Macro F1={macro_f1:.4f}  "
          f"F1_Critical={critical_f1:.4f}  Recall_Critical={critical_rc:.4f}")

print(f"\n  -- CV Summary -------------------------------------------")
print(f"  Macro F1        : {np.mean(fold_macro_f1):.4f} +/- {np.std(fold_macro_f1):.4f}")
print(f"  F1_Critical     : {np.mean(fold_critical_f1):.4f} +/- {np.std(fold_critical_f1):.4f}")
print(f"  Recall_Critical : {np.mean(fold_critical_rc):.4f} +/- {np.std(fold_critical_rc):.4f}")

# --------------------------------------------─
# 6. FINAL TRAINING ON FULL DATASET
# --------------------------------------------─
print("\n[5/7] Training final model on full dataset ...")

X_full_res, y_full_res = smote.fit_resample(X, y)

final_model = lgb.LGBMClassifier(**lgb_params)
final_model.fit(
    X_full_res, y_full_res,
    callbacks=[lgb.log_evaluation(-1)]
)

# Evaluate on full training data
y_pred_full = final_model.predict(X)
print("\n  -- Full-Dataset Metrics ---------------------------------")
print(classification_report(y, y_pred_full,
                             target_names=["Clear", "Low Risk", "Critical"],
                             digits=4))
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(y, y_pred_full)
print(f"Training accuracy: {train_acc:.4f}")

# optional: save to a text file
with open("training_metrics.txt", "w") as f:
    f.write(f"Training accuracy: {train_acc:.4f}\n")


print("  Confusion Matrix:")
cm = confusion_matrix(y, y_pred_full)
cm_df = pd.DataFrame(cm,
                     index=["Actual Clear", "Actual Low Risk", "Actual Critical"],
                     columns=["Pred Clear", "Pred Low Risk", "Pred Critical"])
print(cm_df.to_string())

# Save model + encoders
joblib.dump(final_model, MODEL_PATH)
joblib.dump(encoders,    ENC_PATH)
print(f"\n  Saved {MODEL_PATH} and {ENC_PATH}")

# ---------------------------------------------
# 7. RISK SCORE + SHAP EXPLAINABILITY
# ---------------------------------------------
print("\n[6/7] Computing Risk Scores and SHAP explanations ...")

# Predict probabilities
proba = final_model.predict_proba(X)  # columns: Clear, Low Risk, Critical
p_clear    = proba[:, 0]
p_low_risk = proba[:, 1]
p_critical = proba[:, 2]

# ── LGBM Probability Risk Score ───────────────────────────────────────
risk_raw = 0.0 * p_clear + 0.35 * p_low_risk + 1.0 * p_critical
risk_score_arr = (risk_raw * 100).clip(0, 100).round(2)

def assign_risk_level(score):
    if score >= CRITICAL_THRESH:
        return "Critical"
    elif score >= LOW_RISK_THRESH:
        return "Low Risk"
    else:
        return "Clear"

df["Risk_Score"] = risk_score_arr
df["Risk_Level"] = [assign_risk_level(s) for s in df["Risk_Score"]]

print(f"      Risk Level distribution:\n{df['Risk_Level'].value_counts()}")

# SHAP explanations - use TreeExplainer on a sample for speed
print("      Computing SHAP values (this may take a minute) ...")
explainer    = shap.TreeExplainer(final_model)
shap_values  = explainer.shap_values(X)  # list of 3 arrays [n x features]

# For each sample, take SHAP values from the highest-probability class
predicted_class = final_model.predict(X)
shap_for_pred   = np.array([
    shap_values[c][i]
    for i, c in enumerate(predicted_class)
])

# Human-readable feature name mapping
FEATURE_LABELS = {
    "weight_diff_abs"       : "weight difference (abs)",
    "weight_diff_pct"       : "weight mismatch %",
    "weight_mismatch_flag"  : "weight mismatch >5%",
    "weight_mismatch_severe": "weight mismatch >15%",
    "weight_ratio"          : "measured/declared weight ratio",
    "log_declared_value"    : "declared value",
    "log_declared_weight"   : "declared weight",
    "log_measured_weight"   : "measured weight",
    "log_dwell_time"        : "dwell time",
    "value_per_kg"          : "value per kg",
    "log_value_per_kg"      : "value-per-kg (log)",
    "zero_value_flag"       : "zero declared value",
    "hour_of_day"           : "declaration hour",
    "day_of_week"           : "day of week",
    "month"                 : "month",
    "year"                  : "year",
    "hs_chapter"            : "HS code chapter",
    "is_transit"            : "transit regime",
    "importer_risk_rate"    : "importer risk rate",
    "exporter_risk_rate"    : "exporter risk rate",
    "country_risk_rate"     : "origin country risk rate",
    "port_risk_rate"        : "destination port risk rate",
    "shipping_line_risk_rate": "shipping line risk rate",
    "dest_country_risk_rate": "destination country risk rate",
    "route_risk_rate"        : "origin-to-destination route risk",
    "entity_risk_combined"  : "entity risk (combined)",
    "anomaly_score"         : "anomaly score",
    "is_anomaly"            : "anomaly flag",
    "Origin_Country_enc"    : "origin country",
    "Destination_Country_enc":"destination country",
    "Destination_Port_enc"  : "destination port",
    "Shipping_Line_enc"     : "shipping line",
    "Importer_ID_enc"       : "importer ID",
    "Exporter_ID_enc"       : "exporter ID",
}

PLAIN_ENGLISH = {
    # Weight discrepancy
    ("weight_diff_abs",        "HIGH"): "Declared weight differs significantly from physically measured weight",
    ("weight_diff_pct",        "HIGH"): "Declared weight is more than 5% off from the measured weight",
    ("weight_diff_pct",        "LOW" ): "Declared and measured weights are closely aligned",
    ("weight_mismatch_flag",   "HIGH"): "Weight discrepancy exceeds the 5% tolerance threshold",
    ("weight_mismatch_severe", "HIGH"): "Severe weight discrepancy — more than 15% difference detected",
    ("weight_ratio",           "HIGH"): "Measured weight is much higher than what was declared",
    ("weight_ratio",           "LOW" ): "Measured weight is much lower than what was declared",
    # Value & goods
    ("log_declared_value",     "HIGH"): "Declared value is unusually high for this type of goods",
    ("log_declared_value",     "LOW" ): "Declared value appears suspiciously low",
    ("value_per_kg",           "HIGH"): "Value per kilogram is abnormally high — possible mis-classification of goods",
    ("value_per_kg",           "LOW" ): "Value per kilogram is abnormally low — possible under-valuation",
    ("log_value_per_kg",       "HIGH"): "Value-to-weight ratio is unusually high",
    ("log_value_per_kg",       "LOW" ): "Value-to-weight ratio is unusually low",
    ("zero_value_flag",        "HIGH"): "Declared value is zero — shipment may be under-declared",
    ("log_declared_weight",    "HIGH"): "Declared weight is unusually large",
    ("log_declared_weight",    "LOW" ): "Declared weight is unusually small",
    ("log_measured_weight",    "HIGH"): "Physically measured weight is abnormally high",
    ("log_measured_weight",    "LOW" ): "Physically measured weight is abnormally low",
    # Entity risk
    ("importer_risk_rate",      "HIGH"): "This importer has a history of flagged or seized shipments",
    ("importer_risk_rate",      "LOW" ): "This importer has a clean compliance record",
    ("exporter_risk_rate",      "HIGH"): "This exporter has been associated with previous risk incidents",
    ("exporter_risk_rate",      "LOW" ): "This exporter has a low-risk track record",
    ("country_risk_rate",       "HIGH"): "Shipment originates from a country with elevated smuggling risk",
    ("country_risk_rate",       "LOW" ): "Origin country has a low customs risk profile",
    ("port_risk_rate",          "HIGH"): "Destination port has elevated risk and enforcement activity",
    ("port_risk_rate",          "LOW" ): "Destination port has a low risk profile",
    ("shipping_line_risk_rate", "HIGH"): "The shipping line has been linked to previous risk incidents",
    ("shipping_line_risk_rate", "LOW" ): "The shipping line has a clean compliance record",
    ("dest_country_risk_rate",  "HIGH"): "Destination country has elevated risk indicators",
    ("dest_country_risk_rate",  "LOW" ): "Destination country is low-risk",
    ("route_risk_rate",         "HIGH"): "This trade route has a high incidence of flagged shipments",
    ("route_risk_rate",         "LOW" ): "This is a low-risk trade route",
    ("entity_risk_combined",    "HIGH"): "Multiple parties involved (importer, exporter, route) have elevated risk profiles",
    ("entity_risk_combined",    "LOW" ): "All parties involved have low risk profiles",
    # Anomaly
    ("anomaly_score",  "HIGH"): "Shipment shows highly unusual patterns compared to normal traffic",
    ("anomaly_score",  "LOW" ): "Shipment patterns are consistent with normal traffic",
    ("is_anomaly",     "HIGH"): "Shipment has been flagged as a statistical anomaly",
    # Time patterns
    ("log_dwell_time", "HIGH"): "Shipment has been at port for an unusually long time",
    ("log_dwell_time", "LOW" ): "Dwell time is shorter than typical",
    ("hour_of_day",    "HIGH"): "Declaration was submitted at an unusual hour",
    ("day_of_week",    "HIGH"): "Declaration was submitted on an unusual day of the week",
    ("month",          "HIGH"): "Seasonal pattern suggests elevated risk for this period",
    # Goods classification
    ("hs_chapter",  "HIGH"): "Goods are classified under a high-risk HS code category",
    ("hs_chapter",  "LOW" ): "Goods fall under a low-risk HS code category",
    ("is_transit",  "HIGH"): "Shipment is in transit regime, which carries inherently higher risk",
    # Encoded categoricals (fallback labels)
    ("Origin_Country_enc",      "HIGH"): "Origin country is associated with elevated risk",
    ("Destination_Country_enc", "HIGH"): "Destination country is associated with elevated risk",
    ("Destination_Port_enc",    "HIGH"): "Destination port is associated with elevated risk",
    ("Shipping_Line_enc",       "HIGH"): "Shipping line is associated with elevated risk",
    ("Importer_ID_enc",         "HIGH"): "Importer profile indicates elevated risk",
    ("Exporter_ID_enc",         "HIGH"): "Exporter profile indicates elevated risk",
}

def top_shap_reasons(shap_row, feature_names, top_n=3):
    """Return plain-English reasons that PUSHED TOWARD the predicted risk level.
    Only positive-SHAP features are included so explanations always align with
    the risk level and never show contradicting counter-evidence."""
    # Only keep features that increased probability of predicted class
    positive_idx = np.where(shap_row > 0)[0]
    if len(positive_idx) == 0:
        positive_idx = np.arange(len(shap_row))  # fallback: use all
    top_idx = positive_idx[np.argsort(shap_row[positive_idx])[::-1][:top_n]]
    parts = []
    for i in top_idx:
        feat = feature_names[i]
        sentence = PLAIN_ENGLISH.get(
            (feat, "HIGH"), FEATURE_LABELS.get(feat, feat) + " (elevated)"
        )
        parts.append(sentence)
    return "; ".join(parts)


print("      Building explanation summaries ...")
explanations = [
    top_shap_reasons(shap_for_pred[i], FEATURES, top_n=3)
    for i in range(len(df))
]
df["Explanation_Summary"] = explanations

# --------------------------------------------─
# 8. SAVE OUTPUT CSV
# --------------------------------------------─
print(f"\n[7/7] Saving predictions to '{OUT_PATH}' ...")
out_df = df[["Container_ID", "Risk_Score", "Risk_Level", "Explanation_Summary"]].copy()
out_df.to_csv(OUT_PATH, index=False)
print(f"      Saved {len(out_df)} rows to '{OUT_PATH}'")

# Print a few sample predictions
print("\n  Sample predictions (Critical):")
sample = out_df[out_df["Risk_Level"] == "Critical"].head(5)
print(sample.to_string(index=False))

print("\n  Sample predictions (Low Risk):")
sample = out_df[out_df["Risk_Level"] == "Low Risk"].head(3)
print(sample.to_string(index=False))

print("\n  Sample predictions (Clear):")
sample = out_df[out_df["Risk_Level"] == "Clear"].head(3)
print(sample.to_string(index=False))

print("\n" + "=" * 60)
print("Training Complete!")
print(f"  Model saved   : {MODEL_PATH}")
print(f"  Encoders saved: {ENC_PATH}")
print(f"  Predictions   : {OUT_PATH}")
print("=" * 60)
