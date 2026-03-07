"""
SmartContainer Risk Engine - predict.py
========================================
Inference script: loads saved model + encoders and scores a new shipment CSV.

Usage:
    python predict.py --input "Shipment Data.csv"
    python predict.py --input "Shipment Data.csv" --output my_predictions.csv
    python predict.py --input "Shipment Data.csv" --critical-threshold 55

If the input CSV has a 'Clearance_Status' column, evaluation metrics are also printed.
"""

import argparse, warnings, joblib, os
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

# --------------------------------------------─
# CONFIG - must match train_model.py
# --------------------------------------------─
MODEL_PATH       = "risk_model.pkl"
ENC_PATH         = "label_encoders.pkl"
ISO_PATH         = "isolation_forest.pkl"
RATE_TABLES_PATH = "risk_rate_tables.pkl"
DEFAULT_OUT      = "container_risk_predictions.csv"

LABEL_MAP = {"Clear": 0, "Low Risk": 1, "Critical": 2}

FEATURES = [
    "weight_diff_abs", "weight_diff_pct", "weight_mismatch_flag",
    "weight_mismatch_severe", "weight_ratio",
    "log_declared_value", "log_declared_weight", "log_measured_weight",
    "log_dwell_time", "value_per_kg", "log_value_per_kg", "zero_value_flag",
    "hour_of_day", "day_of_week", "month", "year",
    "hs_chapter", "is_transit",
    "importer_risk_rate", "exporter_risk_rate", "country_risk_rate",
    "port_risk_rate", "shipping_line_risk_rate", "dest_country_risk_rate",
    "route_risk_rate", "entity_risk_combined",
    "anomaly_score", "is_anomaly",
    "Origin_Country_enc", "Destination_Country_enc", "Destination_Port_enc",
    "Shipping_Line_enc", "Importer_ID_enc", "Exporter_ID_enc",
]

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
    "route_risk_rate"       : "origin-to-destination route risk",
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

ISO_FEATURES = [
    "log_declared_value", "log_declared_weight", "log_measured_weight",
    "weight_diff_pct", "weight_ratio", "log_dwell_time", "log_value_per_kg",
    "hour_of_day", "hs_chapter", "importer_risk_rate", "exporter_risk_rate",
    "country_risk_rate", "is_transit"
]


# --------------------------------------------─
# HELPERS
# --------------------------------------------─
def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        "Declaration_Date (YYYY-MM-DD)":          "Declaration_Date",
        "Trade_Regime (Import / Export / Transit)": "Trade_Regime",
    }, inplace=True)
    return df


def engineer_features(df, encoders, rate_tables):
    """Apply identical feature engineering as train_model.py."""
    # Date / Time
    df["Declaration_Date"] = pd.to_datetime(df["Declaration_Date"], errors="coerce")
    df["hour_of_day"] = pd.to_datetime(df["Declaration_Time"], format="%H:%M:%S", errors="coerce").dt.hour.fillna(12)
    df["day_of_week"] = df["Declaration_Date"].dt.dayofweek.fillna(0)
    df["month"]       = df["Declaration_Date"].dt.month.fillna(1)
    df["year"]        = df["Declaration_Date"].dt.year.fillna(2020)

    # Weight features
    df["weight_diff_abs"]       = df["Measured_Weight"] - df["Declared_Weight"]
    df["weight_diff_pct"]       = df["weight_diff_abs"] / (df["Declared_Weight"].abs() + 0.001) * 100
    df["weight_mismatch_flag"]  = (df["weight_diff_pct"].abs() > 5).astype(int)
    df["weight_mismatch_severe"]= (df["weight_diff_pct"].abs() > 15).astype(int)
    df["weight_ratio"]          = df["Measured_Weight"] / (df["Declared_Weight"] + 0.001)

    # Value features
    df["value_per_kg"]       = df["Declared_Value"] / (df["Declared_Weight"] + 0.001)
    df["log_declared_value"] = np.log1p(df["Declared_Value"].clip(lower=0))
    df["log_declared_weight"]= np.log1p(df["Declared_Weight"].clip(lower=0))
    df["log_measured_weight"]= np.log1p(df["Measured_Weight"].clip(lower=0))
    df["log_dwell_time"]     = np.log1p(df["Dwell_Time_Hours"].clip(lower=0))
    df["log_value_per_kg"]   = np.log1p(df["value_per_kg"].clip(lower=0))
    df["zero_value_flag"]    = (df["Declared_Value"] == 0).astype(int)

    # HS Chapter
    df["hs_chapter"] = (df["HS_Code"] // 10000).astype(int)

    # Trade regime
    df["is_transit"] = (df["Trade_Regime"].str.strip() == "Transit").astype(int)

    # Entity risk rates (lookup from training tables; fallback to global rate)
    global_rate = 0.01   # approximate training critical rate

    def lookup_rate(col, table_key):
        df["route"] = df["Origin_Country"].astype(str) + "_" + df["Destination_Country"].astype(str)
        lookup_col = "route" if col == "route" else col
        tbl = rate_tables.get(table_key, {})
        return df[lookup_col].map(tbl).fillna(global_rate)

    df["route"] = df["Origin_Country"].astype(str) + "_" + df["Destination_Country"].astype(str)
    df["importer_risk_rate"]     = df["Importer_ID"].map(rate_tables.get("importer_risk_rate", {})).fillna(global_rate)
    df["exporter_risk_rate"]     = df["Exporter_ID"].map(rate_tables.get("exporter_risk_rate", {})).fillna(global_rate)
    df["country_risk_rate"]      = df["Origin_Country"].map(rate_tables.get("country_risk_rate", {})).fillna(global_rate)
    df["port_risk_rate"]         = df["Destination_Port"].map(rate_tables.get("port_risk_rate", {})).fillna(global_rate)
    df["shipping_line_risk_rate"]= df["Shipping_Line"].map(rate_tables.get("shipping_line_risk_rate", {})).fillna(global_rate)
    df["dest_country_risk_rate"] = df["Destination_Country"].map(rate_tables.get("dest_country_risk_rate", {})).fillna(global_rate)
    df["route_risk_rate"]        = df["route"].map(rate_tables.get("route_risk_rate", {})).fillna(global_rate)

    df["entity_risk_combined"] = (
        0.30 * df["importer_risk_rate"] +
        0.20 * df["exporter_risk_rate"] +
        0.15 * df["country_risk_rate"] +
        0.10 * df["port_risk_rate"] +
        0.10 * df["shipping_line_risk_rate"] +
        0.10 * df["dest_country_risk_rate"] +
        0.05 * df["route_risk_rate"]
    )

    # Label encode with fallback for unseen categories
    cat_cols = ["Origin_Country", "Destination_Country", "Destination_Port",
                "Shipping_Line", "Importer_ID", "Exporter_ID", "Trade_Regime"]
    for col in cat_cols:
        le = encoders.get(col)
        if le is None:
            df[col + "_enc"] = 0
            continue
        known = set(le.classes_)
        df[col + "_enc"] = df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )

    return df


def apply_anomaly(df, iso_model):
    df["anomaly_score"] = -iso_model.score_samples(df[ISO_FEATURES])
    df["is_anomaly"]    = (iso_model.predict(df[ISO_FEATURES]) == -1).astype(int)
    return df




def assign_risk_level(score, critical_thresh, low_risk_thresh):
    if score >= critical_thresh:
        return "Critical"
    elif score >= low_risk_thresh:
        return "Low Risk"
    else:
        return "Clear"


PLAIN_ENGLISH = {
    ("weight_diff_abs",        "HIGH"): "Declared weight differs significantly from physically measured weight",
    ("weight_diff_pct",        "HIGH"): "Declared weight is more than 5% off from the measured weight",
    ("weight_diff_pct",        "LOW" ): "Declared and measured weights are closely aligned",
    ("weight_mismatch_flag",   "HIGH"): "Weight discrepancy exceeds the 5% tolerance threshold",
    ("weight_mismatch_severe", "HIGH"): "Severe weight discrepancy — more than 15% difference detected",
    ("weight_ratio",           "HIGH"): "Measured weight is much higher than what was declared",
    ("weight_ratio",           "LOW" ): "Measured weight is much lower than what was declared",
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
    ("anomaly_score",  "HIGH"): "Shipment shows highly unusual patterns compared to normal traffic",
    ("anomaly_score",  "LOW" ): "Shipment patterns are consistent with normal traffic",
    ("is_anomaly",     "HIGH"): "Shipment has been flagged as a statistical anomaly",
    ("log_dwell_time", "HIGH"): "Shipment has been at port for an unusually long time",
    ("log_dwell_time", "LOW" ): "Dwell time is shorter than typical",
    ("hour_of_day",    "HIGH"): "Declaration was submitted at an unusual hour",
    ("day_of_week",    "HIGH"): "Declaration was submitted on an unusual day of the week",
    ("month",          "HIGH"): "Seasonal pattern suggests elevated risk for this period",
    ("hs_chapter",  "HIGH"): "Goods are classified under a high-risk HS code category",
    ("hs_chapter",  "LOW" ): "Goods fall under a low-risk HS code category",
    ("is_transit",  "HIGH"): "Shipment is in transit regime, which carries inherently higher risk",
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


# --------------------------------------------─
# MAIN
# --------------------------------------------─
def main():
    parser = argparse.ArgumentParser(description="SmartContainer Risk Engine - Predict")
    parser.add_argument("--input",              required=True,              help="Path to shipment CSV")
    parser.add_argument("--output",             default=DEFAULT_OUT,        help="Output CSV path")
    parser.add_argument("--critical-threshold", type=float, default=55.0,  help="Risk score cutoff for Critical (0-100)")
    parser.add_argument("--low-risk-threshold", type=float, default=22.0,  help="Risk score cutoff for Low Risk (0-100)")
    parser.add_argument("--no-shap",            action="store_true",       help="Skip SHAP (faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("SmartContainer Risk Engine - Predict")
    print("=" * 60)

    # Load artefacts
    print(f"\nLoading model artefacts ...")
    for p in [MODEL_PATH, ENC_PATH, ISO_PATH, RATE_TABLES_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}. Run train_model.py first.")

    model       = joblib.load(MODEL_PATH)
    encoders    = joblib.load(ENC_PATH)
    iso_model   = joblib.load(ISO_PATH)
    rate_tables = joblib.load(RATE_TABLES_PATH)

    # Load & preprocess test data
    print(f"Loading test data from '{args.input}' ...")
    df = load_and_clean(args.input)
    print(f"  Shape: {df.shape}")

    df = engineer_features(df, encoders, rate_tables)
    df = apply_anomaly(df, iso_model)

    X = df[FEATURES].copy()

    # Predict
    proba      = model.predict_proba(X)
    pred_class = model.predict(X)
    risk_raw   = 0.35 * proba[:, 1] + 1.0 * proba[:, 2]
    risk_score = (risk_raw * 100).clip(0, 100).round(2)

    df["Risk_Score"] = risk_score
    df["Risk_Level"] = [
        assign_risk_level(s, args.critical_threshold, args.low_risk_threshold)
        for s in df["Risk_Score"]
    ]

    print(f"\nRisk Level distribution:\n{df['Risk_Level'].value_counts()}")

    # SHAP
    if not args.no_shap:
        print("\nComputing SHAP explanations ...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Handle both ndarray (n_samples, n_features, n_classes) [newer shap]
        # and list of (n_samples, n_features) arrays [older shap]
        sv = np.array(shap_values)
        if sv.ndim == 3:
            shap_pred = np.array([sv[i, :, c] for i, c in enumerate(pred_class)])
        else:
            shap_pred = np.array([sv[c][i] for i, c in enumerate(pred_class)])
        explanations = [top_shap_reasons(shap_pred[i], FEATURES) for i in range(len(df))]
    else:
        explanations = ["(SHAP disabled)"] * len(df)

    df["Explanation_Summary"] = explanations

    # Evaluation (if ground truth present)
    if "Clearance_Status" in df.columns:
        from sklearn.metrics import (
            f1_score, recall_score, classification_report, confusion_matrix
        )
        y_true = df["Clearance_Status"].map(LABEL_MAP)
        y_pred_label = df["Risk_Level"].map(LABEL_MAP)

        print("\n-- Evaluation Metrics ------------------------------")
        print(classification_report(y_true, y_pred_label,
                                    target_names=["Clear", "Low Risk", "Critical"],
                                    digits=4))
        cm = confusion_matrix(y_true, y_pred_label)
        cm_df = pd.DataFrame(cm,
                             index=["Actual Clear", "Actual Low Risk", "Actual Critical"],
                             columns=["Pred Clear", "Pred Low Risk", "Pred Critical"])
        print("Confusion Matrix:")
        print(cm_df.to_string())

        macro_f1    = f1_score(y_true, y_pred_label, average="macro")
        critical_f1 = f1_score(y_true, y_pred_label, average=None, labels=[2])[0]
        critical_rc = recall_score(y_true, y_pred_label, average=None, labels=[2])[0]
        weighted_f1 = f1_score(y_true, y_pred_label, average="weighted")

        print(f"\nPrimary   - Macro F1      : {macro_f1:.4f}")
        print(f"Secondary - F1_Critical   : {critical_f1:.4f}")
        print(f"Secondary - Recall_Critical: {critical_rc:.4f}")
        print(f"Secondary - Weighted F1   : {weighted_f1:.4f}")

    # Save output
    out_df = df[["Container_ID", "Risk_Score", "Risk_Level", "Explanation_Summary"]].copy()
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(out_df)} predictions -to- '{args.output}'")
    print("\nSample (Critical):")
    print(out_df[out_df["Risk_Level"] == "Critical"].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
