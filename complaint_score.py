"""
complaint_score_v3.py
Fixes header row detection for Complaint sheet
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

excel_path = r"C:\Users\Mounika.K\Downloads\SAP Hackathon Field Mapping and Example (1).xlsx"
sheet_compl = "Complainnt Data"
REF_DATE = pd.Timestamp("2025-10-24")
HALF_LIFE_COMPL = 120.0

# ---------------- UTILITIES ----------------
def time_decay(age_days, half_life):
    if pd.isna(age_days):
        return 0.0
    return np.exp(-np.log(2) * age_days / half_life)

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

# ---------------- LOAD WITH HEADER DETECTION ----------------
print("📘 Reading Complaint sheet...")
first = pd.read_excel(excel_path, sheet_name=sheet_compl, header=None)
header_row = first[first.astype(str).apply(lambda x: x.str.contains("Date", case=False, na=False)).any(axis=1)].index.min()
if pd.isna(header_row):
    header_row = 0
print(f"✅ Detected header row: {header_row}")

df = pd.read_excel(excel_path, sheet_name=sheet_compl, header=header_row)
df.columns = df.columns.astype(str).str.replace("\n", " ").str.strip()
print("✅ Columns detected:", list(df.columns))

# ---------------- COMPUTE SCORES ----------------
req_cols = ["Date", "batch", "Outcome category", "Complaint Outcome", "RCA Category"]
for col in req_cols:
    if col not in df.columns:
        print(f"⚠️ Missing column in Complaint sheet: {col}")

df["_sev"] = df.get("Outcome category", "").apply(severity_weight)
df["_recall"] = df.get("Complaint Outcome", "").apply(recall_multi)
df["_cause"] = df.get("RCA Category", "").apply(cause_weight)

df["_date"] = pd.to_datetime(df.get("Date"), errors="coerce", dayfirst=True)
df["_age_days"] = (REF_DATE - df["_date"]).dt.days.clip(lower=0)
df["_time_decay"] = df["_age_days"].apply(lambda d: time_decay(d, HALF_LIFE_COMPL))

df["complain_score"] = df["_sev"] * df["_recall"] * df["_cause"] * df["_time_decay"]

batch_scores = (
    df.groupby("batch", dropna=False)
      .agg(mean_compl_score=("complain_score", "mean"))
      .reset_index()
)

batch_scores.to_csv("complaint_scores_batches.csv", index=False)
print("✅ Complaint Scores saved → complaint_scores_batches.csv")
print(batch_scores.head(10))
