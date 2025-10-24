import pandas as pd
import numpy as np

# Path and sheet name
excel_path = r"C:\Users\Mounika.K\Downloads\SAP Hackathon Field Mapping and Example (1).xlsx"
sheet_ipc = "Order IPC Insp+Sample Results"

# 🧠 Read using the *second row as header* (row index 1 → second row)
df = pd.read_excel(excel_path, sheet_name=sheet_ipc, header=1)

# 🧹 Clean column names
df.columns = (
    df.columns.astype(str)
    .str.replace(r'[\n\r"]', '', regex=True)
    .str.strip()
)

print("✅ Cleaned column names:")
print(df.columns.tolist())

# Detect the key columns dynamically (in case names differ slightly)
ear_col = [c for c in df.columns if "Ear Width" in c][0]
pen_col = [c for c in df.columns if "Penetration" in c and "Value" in c][0]
batch_col = [c for c in df.columns if "Batch" in c][0]

print(f"\n✅ Using columns:\nEar = {ear_col}\nPenetration = {pen_col}\nBatch = {batch_col}")

# Convert to numeric
df["_ear"] = pd.to_numeric(df[ear_col], errors="coerce")
df["_penetration"] = pd.to_numeric(df[pen_col], errors="coerce")

# Spec limits
ear_min, ear_max = 0.1095, 0.1115
pen_min, pen_max = 65, 99

# Compute fail flag using OR condition
df["Fail_Flag"] = (
    (df["_ear"] < ear_min) | (df["_ear"] > ear_max) |
    (df["_penetration"] < pen_min) | (df["_penetration"] > pen_max)
)

# Batch-level fail ratio
fail_summary = (
    df.groupby(batch_col)["Fail_Flag"]
      .mean()
      .reset_index(name="Inspection_Fail_Ratio")
)

# Save to CSV
fail_summary.to_csv("inspection_fail_scores_batches.csv", index=False)

print("\n📊 Inspection Fail Ratios per Batch:")
print(fail_summary)

print("\n✅ Saved → inspection_fail_scores_batches.csv")
