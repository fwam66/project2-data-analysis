# Decision Tree with Confusion Matrix visualisations
# Predicts trip main mode from household income group and home LGA

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


CSV_PATH = "merged_data.csv"
TARGET_COL = "mainmode"               
INCOME_COL = "weekly_hhinc_group"
LGA_COL = "homelga"
WEIGHT_COL = "trippoststratweight"

# Tree visuals
MAX_DEPTH_FOR_PLOT = 4     # keeps the plot readable
MIN_SAMPLES_LEAF = 50      # regularization for stability

# Load data
df = pd.read_csv(CSV_PATH)

# Keep necessary rows
req_cols = [TARGET_COL, INCOME_COL, LGA_COL]
missing = [c for c in req_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in {CSV_PATH}: {missing}")

df = df[df[TARGET_COL].notna() & df[INCOME_COL].notna() & df[LGA_COL].notna()].copy()

# Ensure types
df[INCOME_COL] = df[INCOME_COL].astype(int)
df[LGA_COL] = df[LGA_COL].astype(str)
df[TARGET_COL] = df[TARGET_COL].astype(str)

# Features & target
X = df[[INCOME_COL, LGA_COL]]
y = df[TARGET_COL]

# Weights
has_weight = WEIGHT_COL in df.columns and df[WEIGHT_COL].notna().any()

# LGA was one-hot encoded, pass income through
pre = ColumnTransformer(
    transformers=[
        ("ohe_lga", OneHotEncoder(handle_unknown="ignore"), [LGA_COL]),
    ],
    remainder="passthrough",
)

# Classifier
clf = DecisionTreeClassifier(
    max_depth=MAX_DEPTH_FOR_PLOT,      # bounded for cleaner visualisation
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight="balanced",           # handles class imbalance if no survey weights
    random_state=42,
)

pipe = Pipeline([("pre", pre), ("clf", clf)])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

sample_weight_train = None
if has_weight:
    # align weights to training rows
    sample_weight_train = df.loc[X_train.index, WEIGHT_COL].values

# Fit
pipe.fit(X_train, y_train, **({"clf__sample_weight": sample_weight_train} if has_weight else {}))

# Predict
y_pred = pipe.predict(X_test)

# Confusion Matrix plots
labels = np.unique(y)  # ensure stable order across plots

# Raw counts confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure()
disp.plot(values_format="d")  # default colormap; no explicit colors
plt.title("Confusion Matrix (Counts)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)

# Normalized confusion matrix
cm_norm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
plt.figure()
disp_norm.plot(values_format=".2f")  # default colormap; no explicit colors
plt.title("Confusion Matrix (Normalized)")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", dpi=300)

print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=3))