# split_train_test.py
# Simple, group-aware 80/20 splits for:
#  A) Classification (target = mainmode)
#  B) Regression    (target = triptime)
# Uses GroupShuffleSplit with groups = hhid.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

# --------------------
# Params (tweak here)
# --------------------
INPUT_FILE   = "merged_data.csv"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
GROUP_COL    = "hhid"   # grouping by household

os.makedirs("outputs", exist_ok=True)

# -------------
# Load once
# -------------
df = pd.read_csv(INPUT_FILE)

if GROUP_COL not in df.columns:
    raise ValueError(f"Group column '{GROUP_COL}' not found in {INPUT_FILE}")

# =========================================================
# (A) CLASSIFICATION: mainmode
# =========================================================
cls_df = df.dropna(subset=["mainmode"]).copy()
y_cls = cls_df["mainmode"].astype(str).values
idx   = np.arange(len(cls_df))

sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr_idx, te_idx = next(sss.split(idx, y_cls))

cls_train = cls_df.iloc[tr_idx].copy()
cls_test  = cls_df.iloc[te_idx].copy()
'''
subsample_hired = df[df['mainmode'] == 'Hired'].sample(n=0, replace=True, random_state=20008)   
subsample_other = df[df['mainmode'] == 'Other'].sample(n=0, replace=True, random_state=20008)   

balanced_df = cls_test[(cls_test['mainmode'] != 'Hired') | (cls_test['mainmode'] != 'Other')]
balanced_df = pd.concat([
    balanced_df, subsample_hired, subsample_other, 
    ])
'''

print(cls_test['mainmode'].value_counts())

cls_train.to_csv(os.path.join("outputs", "classification_train.csv"), index=False)
cls_test.to_csv(os.path.join("outputs", "classification_test.csv"), index=False)

print("[CLASSIFICATION] Splitter: StratifiedShuffleSplit")
print("[CLASSIFICATION] Class % in TRAIN:\n",
      (cls_train["mainmode"].value_counts(normalize=True)*100).round(2))
print("[CLASSIFICATION] Class % in TEST:\n",
      (cls_test["mainmode"].value_counts(normalize=True)*100).round(2))

# ======================================================
# (B) REGRESSION: triptime (Group-aware split by hhid)
# ======================================================
reg_df = df.dropna(subset=["triptime", GROUP_COL]).copy()
reg_df["triptime"] = pd.to_numeric(reg_df["triptime"], errors="coerce")
reg_df = reg_df.dropna(subset=["triptime"])

idx_reg = np.arange(len(reg_df))
groups_reg = reg_df[GROUP_COL].astype(str).values

gss_reg = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr_idx_reg, te_idx_reg = next(gss_reg.split(idx_reg, groups=groups_reg))

reg_train = reg_df.iloc[tr_idx_reg].copy()
reg_test  = reg_df.iloc[te_idx_reg].copy()

reg_train.to_csv(os.path.join("outputs", "regression_train.csv"), index=False)
reg_test.to_csv(os.path.join("outputs", "regression_test.csv"), index=False)

print("[REGRESSION] Splitter: GroupShuffleSplit")
print("[REGRESSION] triptime mean (train/test):",
      round(reg_train["triptime"].mean(), 2), "/",
      round(reg_test["triptime"].mean(), 2))

print(f"\nSaved train/test CSVs to ./{"outputs"}/")
