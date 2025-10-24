"""
capa_score_batches.py

Computes CAPA-based quality score per batch
using CAPA-QN data (from Excel sheet).
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
excel_path = r"C:\Users\Mounika.K\Downloads\SAP Hackathon Field Mapping and Example (1).xlsx"
sheet_capa = "CAPA-QN data"

REF_DATE = pd.Timestamp("2025-10-24")

STATUS_WEIGHT = {"accepted": 1.0, "in progress": 0.8, "rejected": 0.6, "closed": 0.25}
CAUSE_WEIGHT_MAP = {"vendor": 1.0, "component": 0.8, "production": 0.7}
HALF_LIFE_CAPA = 90.0  # decay over time


# ---------------- UTILITIES ----------------
def time_decay(age_days, half_life):
    """Applies exponential time decay based on half-life."""
    if pd.isna(age_days):
        return 0.0
    return np.exp(-np.log(2) * age_days / half_life)

def map_status(s):
    if pd.isna(s): return 0.6
    return STATUS_WEIGHT.get(str(s).strip().lower(), 0.6)

def map_cause(s):
    if pd.isna(s): return 0.7
    for k, v in CAUSE_WEIGHT_MAP.items():
        if k in str(s).lower():
            return v
    return 0.7


# ---------------- LOAD DATA ----------------
print("📘 Reading CAPA sheet...")
xls = pd.ExcelFile(excel_path)
capa_df = pd.read_excel(xls, sheet_capa)
capa_df.columns = capa_df.columns.astype(str).str.replace('\n', ' ').str.strip()

print(f"Columns detected: {list(capa_df.columns)}")


# ---------------- CAPA SCORE COMPUTATION ----------------
def compute_capa_by_batch(df):
    if df.empty:
        print("⚠️ CAPA sheet is empty.")
        return pd.DataFrame(columns=["batch", "mean_capa_score"])
    
    date_col = "Date"
    batch_col = "Batch"
    cause_col = "PR Root cause"
    status_col = "PR Disposition"

    for col in [date_col, batch_col, cause_col, status_col]:
        if col not in df.columns:
            print(f"⚠️ Missing column in CAPA sheet: {col}")

    df["_status_w"] = df.get(status_col, "").apply(map_status)
    df["_cause_w"] = df.get(cause_col, "").apply(map_cause)
    df["_date"] = pd.to_datetime(df.get(date_col), errors="coerce")

    # Age & decay
    df["_age_days"] = (REF_DATE - df["_date"]).dt.days.clip(lower=0)
    df["_time_decay"] = df["_age_days"].apply(lambda d: time_decay(d, HALF_LIFE_CAPA))

    # CAPA score = f(status, cause, recency)
    df["capa_score"] = df["_status_w"] * df["_cause_w"] * df["_time_decay"]

    capa_agg = (
        df.groupby(batch_col, dropna=False)
          .agg(mean_capa_score=("capa_score", "mean"))
          .reset_index()
          .rename(columns={batch_col: "batch"})
    )

    print(f"✅ CAPA batches aggregated: {len(capa_agg)} rows")
    return capa_agg


# ---------------- RUN ----------------
capa_batch_scores = compute_capa_by_batch(capa_df)

# ---------------- SAVE ----------------
capa_batch_scores.to_csv("capa_scores_batches.csv", index=False)
print("✅ Saved CAPA batch-level scores → capa_scores_batches.csv")
print(capa_batch_scores.head(10))
