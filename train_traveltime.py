# train_taskB_linear.py
# Linear Regression only (two variants): B1 with cumdist, B2 without cumdist

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline          # <-- FIX: import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_CSV = os.path.join("outputs", "regression_train.csv")
TEST_CSV  = os.path.join("outputs", "regression_test.csv")
os.makedirs("outputs", exist_ok=True)

# -----------------------
# Load & basic casting
# -----------------------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

target = "triptime"
num_common = ["weekly_hhinc_group", "starthour", "totalvehs"]
cat_feats  = ["homelga", "dayType"]

for c in num_common + ["cumdist", target]:
    if c in train.columns: train[c] = pd.to_numeric(train[c], errors="coerce")
    if c in test.columns:  test[c]  = pd.to_numeric(test[c],  errors="coerce")

train = train.dropna(subset=[target]).copy()
test  = test.dropna(subset=[target]).copy()

# ============================================================
# With distance (cumdist)
# ============================================================
num_feats_wd = [c for c in (num_common + ["cumdist"]) if c in train.columns]
cat_feats_wd = [c for c in cat_feats if c in train.columns]

X_train = train[num_feats_wd + cat_feats_wd].copy()
y_train = train[target].astype(float).copy()
X_test  = test[num_feats_wd + cat_feats_wd].copy()
y_test  = test[target].astype(float).copy()

pre_wd = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_feats_wd),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore"))
    ]), cat_feats_wd),
])

Xtr = pre_wd.fit_transform(X_train)
Xte = pre_wd.transform(X_test)

lin_wd = LinearRegression()
lin_wd.fit(Xtr, y_train)
pred_wd = lin_wd.predict(Xte)

mae  = mean_absolute_error(y_test, pred_wd)
mse  = mean_squared_error(y_test, pred_wd)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, pred_wd)
print(f"\n[with_distance] LinearRegression  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

plt.figure()
plt.scatter(y_test, pred_wd, s=8, alpha=0.6)
lo, hi = np.nanmin([y_test.min(), pred_wd.min()]), np.nanmax([y_test.max(), pred_wd.max()])
plt.plot([lo, hi], [lo, hi])
plt.xlabel("Actual triptime (min)")
plt.ylabel("Predicted triptime (min)")
plt.title("Linear Regression (with cumulative distance)")
plt.tight_layout()
plt.savefig(os.path.join("outputs", "Linear_Regression_with_distance.png"), dpi=200)
plt.close()

# ============================================================
# Without distance (no cumdist)
# ============================================================
num_feats_nd = [c for c in num_common if c in train.columns]
cat_feats_nd = [c for c in cat_feats if c in train.columns]

X_train = train[num_feats_nd + cat_feats_nd].copy()
y_train = train[target].astype(float).copy()
X_test  = test[num_feats_nd + cat_feats_nd].copy()
y_test  = test[target].astype(float).copy()

pre_nd = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_feats_nd),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore"))
    ]), cat_feats_nd),
])

Xtr = pre_nd.fit_transform(X_train)
Xte = pre_nd.transform(X_test)

lin_nd = LinearRegression()
lin_nd.fit(Xtr, y_train)
pred_nd = lin_nd.predict(Xte)

# --- no distance eval ---
mae  = mean_absolute_error(y_test, pred_nd)
mse  = mean_squared_error(y_test, pred_nd)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, pred_nd)
print(f"[without_distance]   LinearRegression  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

plt.figure()
plt.scatter(y_test, pred_nd, s=8, alpha=0.6)
lo, hi = np.nanmin([y_test.min(), pred_nd.min()]), np.nanmax([y_test.max(), pred_nd.max()])
plt.plot([lo, hi], [lo, hi])
plt.xlabel("Actual triptime (min)")
plt.ylabel("Predicted triptime (min)")
plt.title("Linear Regression")
plt.tight_layout()
plt.savefig(os.path.join("outputs", "Linear_Regression.png"), dpi=200)
plt.close()
