"""
quality_pipeline_singleexcel_rf.py

Full End-to-End Quality Pipeline
---------------------------------------------------
1️⃣ Compute inspection, CAPA, and complaint scores
2️⃣ Combine into ONE Excel file → 'quality_scores_all.xlsx'
3️⃣ Train Random Forest models:
     - Regression → Predict quality score
     - Classification → Predict complaint risk
4️⃣ Save model outputs to Excel → 'final_quality_predictions.xlsx'
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
excel_path = r"C:\Users\Mounika.K\Downloads\SAP Hackathon Field Mapping and Example (1).xlsx"
sheet_ipc = "Order IPC Insp+Sample Results"
sheet_capa = "CAPA-QN data"
sheet_compl = "Complainnt Data"

#REF_DATE = pd.Timestamp("2025-10-24")
from datetime import datetime
REF_DATE = pd.Timestamp(datetime.now().date())

# ---------------- UTILITIES ----------------
def time_decay(age_days, half_life):
    if pd.isna(age_days):
        return 0.0
    return np.exp(-np.log(2) * age_days / half_life)


# =========================================================
# 1️⃣ INSPECTION SCORE
# =========================================================
def compute_inspection_scores():
    df = pd.read_excel(excel_path, sheet_name=sheet_ipc, header=1)
    df.columns = df.columns.astype(str).str.replace(r'[\n\r"]', '', regex=True).str.strip()

    ear_col = [c for c in df.columns if "Ear Width" in c][0]
    pen_col = [c for c in df.columns if "Penetration" in c and "Value" in c][0]
    batch_col = [c for c in df.columns if "Batch" in c][0]

    df["_ear"] = pd.to_numeric(df[ear_col], errors="coerce")
    df["_pen"] = pd.to_numeric(df[pen_col], errors="coerce")

    ear_min, ear_max = 0.1095, 0.1115
    pen_min, pen_max = 65, 99

    df["Fail_Flag"] = (
        (df["_ear"] < ear_min) | (df["_ear"] > ear_max) |
        (df["_pen"] < pen_min) | (df["_pen"] > pen_max)
    )

    insp = (
        df.groupby(batch_col)["Fail_Flag"]
          .mean()
          .reset_index(name="Inspection_Fail_Ratio")
    )
    insp = insp.rename(columns={batch_col: "batch"})
    insp["insp_score"] = 1 - insp["Inspection_Fail_Ratio"]
    return insp


# =========================================================
# 2️⃣ CAPA SCORE
# =========================================================
def compute_capa_scores():
    STATUS_WEIGHT = {"accepted": 1.0, "in progress": 0.8, "rejected": 0.6, "closed": 0.25}
    CAUSE_WEIGHT_MAP = {"vendor": 1.0, "component": 0.8, "production": 0.7}
    HALF_LIFE_CAPA = 90.0

    capa_df = pd.read_excel(excel_path, sheet_name=sheet_capa)
    capa_df.columns = capa_df.columns.astype(str).str.replace("\n", " ").str.strip()

    def map_status(s):
        if pd.isna(s): return 0.6
        return STATUS_WEIGHT.get(str(s).strip().lower(), 0.6)

    def map_cause(s):
        if pd.isna(s): return 0.7
        for k, v in CAUSE_WEIGHT_MAP.items():
            if k in str(s).lower():
                return v
        return 0.7

    capa_df["_status_w"] = capa_df.get("PR Disposition", "").apply(map_status)
    capa_df["_cause_w"] = capa_df.get("PR Root cause", "").apply(map_cause)
    capa_df["_date"] = pd.to_datetime(capa_df.get("Date"), errors="coerce")
    capa_df["_age_days"] = (REF_DATE - capa_df["_date"]).dt.days.clip(lower=0)
    capa_df["_time_decay"] = capa_df["_age_days"].apply(lambda d: time_decay(d, HALF_LIFE_CAPA))
    capa_df["capa_score"] = capa_df["_status_w"] * capa_df["_cause_w"] * capa_df["_time_decay"]

    capa = (
        capa_df.groupby("Batch", dropna=False)
               .agg(capa_score=("capa_score", "mean"))
               .reset_index()
               .rename(columns={"Batch": "batch"})
    )
    return capa


# =========================================================
# 3️⃣ COMPLAINT SCORE
# =========================================================
def compute_complaint_scores():
    HALF_LIFE_COMPL = 120.0

    first = pd.read_excel(excel_path, sheet_name=sheet_compl, header=None)
    header_row = first[first.astype(str).apply(lambda x: x.str.contains("Date", case=False, na=False)).any(axis=1)].index.min()
    if pd.isna(header_row):
        header_row = 0

    df = pd.read_excel(excel_path, sheet_name=sheet_compl, header=header_row)
    df.columns = df.columns.astype(str).str.replace("\n", " ").str.strip()

    def severity_weight(val):
        if pd.isna(val): return 0.5
        return 1.0 if "major" in str(val).lower() else 0.5

    def recall_multi(val):
        if pd.isna(val): return 1.0
        return 1.2 if "recall" in str(val).lower() else 1.0

    def cause_weight(val):
        if pd.isna(val): return 0.6
        v = str(val).lower()
        if "product" in v or "production" in v:
            return 0.7
        if "component" in v:
            return 0.8
        return 0.6

    df["_sev"] = df.get("Outcome category", "").apply(severity_weight)
    df["_recall"] = df.get("Complaint Outcome", "").apply(recall_multi)
    df["_cause"] = df.get("RCA Category", "").apply(cause_weight)

    df["_date"] = pd.to_datetime(df.get("Date"), errors="coerce", dayfirst=True)
    df["_age_days"] = (REF_DATE - df["_date"]).dt.days.clip(lower=0)
    df["_time_decay"] = df["_age_days"].apply(lambda d: time_decay(d, HALF_LIFE_COMPL))

    df["compl_score"] = df["_sev"] * df["_recall"] * df["_cause"] * df["_time_decay"]

    compl = (
        df.groupby("batch", dropna=False)
          .agg(compl_score=("compl_score", "mean"))
          .reset_index()
    )
    return compl


# =========================================================
# 4️⃣ COMBINE INTO SINGLE EXCEL FILE
# =========================================================
def create_combined_excel():
    insp = compute_inspection_scores()
    capa = compute_capa_scores()
    compl = compute_complaint_scores()

    combined = insp.merge(capa, on="batch", how="outer").merge(compl, on="batch", how="outer")
    combined.fillna(0, inplace=True)

    combined["total_score"] = (
        0.45 * combined["insp_score"] +
        0.35 * combined["capa_score"] +
        0.20 * combined["compl_score"]
    )

    # Save each as sheet
    with pd.ExcelWriter("quality_scores_all.xlsx") as writer:
        insp.to_excel(writer, sheet_name="Inspection_Score", index=False)
        capa.to_excel(writer, sheet_name="CAPA_Score", index=False)
        compl.to_excel(writer, sheet_name="Complaint_Score", index=False)
        combined.to_excel(writer, sheet_name="Combined_Scores", index=False)

    print("✅ Combined Excel saved → quality_scores_all.xlsx")
    return combined


# =========================================================
# 5️⃣ TRAIN MODELS + SAVE TO EXCEL
# =========================================================
def train_models(df):
    X = df[["insp_score", "capa_score", "compl_score"]]
    y = df["total_score"]

    # Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    df["predicted_quality_score"] = reg.predict(X)

    # Classification
    df["complaint_flag"] = (df["compl_score"] > df["compl_score"].median()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, df["complaint_flag"], test_size=0.3, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_class)

    df["complaint_probability"] = clf.predict_proba(X)[:, 1]

    def traffic_light(prob):
        if prob < 0.33:
            return "🟢 Green"
        elif prob < 0.66:
            return "🟡 Yellow"
        else:
            return "🔴 Red"

    df["traffic_light"] = df["complaint_probability"].apply(traffic_light)

    # Save final output to Excel
    with pd.ExcelWriter("final_quality_predictions.xlsx") as writer:
        df.to_excel(writer, sheet_name="Predictions", index=False)

        summary = pd.DataFrame({
            "Metric": ["R² (Regression)", "Accuracy (Classification)", "Timestamp"],
            "Value": [r2, acc, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        summary.to_excel(writer, sheet_name="Model_Summary", index=False)

    print("\n✅ Model results saved → final_quality_predictions.xlsx")
    print(f"📈 Regression R²: {r2:.3f} | 🤖 Classification Accuracy: {acc:.3f}")
    return df


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("🚀 Running full pipeline...\n")
    combined_df = create_combined_excel()
    train_models(combined_df)
    print("\n🎯 Pipeline complete — Excel files generated successfully!")
