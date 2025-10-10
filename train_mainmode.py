# train_taskA.py
# Minimal Task A trainer: Decision Tree + k-NN
# - reads outputs/classification_train.csv / classification_test.csv
# - prints classification_report
# - saves confusion matrix images (PNG)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


TRAIN_CSV = os.path.join("outputs", "classification_train_balanced.csv")
TEST_CSV  = os.path.join("outputs", "classification_test.csv")
os.makedirs("outputs", exist_ok=True)

# -----------------------
# Load
# -----------------------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

# Features (simple; no leakage)
num_feats = ["weekly_hhinc_group", "starthour", "arrhour", "totalvehs"]
cat_feats = ["homelga", "dayType"]
target = "mainmode"

# Cast numerics quietly (turns bad tokens into NaN for imputation)
for c in num_feats + ["triptime", "cumdist"]:
    if c in train.columns: train[c] = pd.to_numeric(train[c], errors="coerce")
    if c in test.columns:  test[c]  = pd.to_numeric(test[c],  errors="coerce")

X_train = train[num_feats + cat_feats].copy()
y_train = train[target].astype(str)
X_test  = test[num_feats + cat_feats].copy()
y_test  = test[target].astype(str)

# -----------------------
# Preprocess (fit on train)
# -----------------------
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_feats),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_feats),
])

Xtr = preprocess.fit_transform(X_train)
Xte = preprocess.transform(X_test)

# =======================
# Model 1: Decision Tree
# =======================
tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=8,
    min_samples_leaf=50
)
tree.fit(Xtr, y_train)
y_pred_tree = tree.predict(Xte)

print("\n=== DecisionTree — classification_report ===")
print(classification_report(y_test, y_pred_tree, digits=3, zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_tree, normalize="true", values_format=".2f"
)
plt.title("DecisionTree — Confusion Matrix (normalized=true)")
plt.tight_layout()
plt.savefig(os.path.join("outputs", "DecisionTree_confusion_matrix.png"), dpi=200)
plt.close()

# ===========
# Model 2: k-NN
# ===========
knn = Pipeline([
    ("scale", MaxAbsScaler()),  # works well with sparse one-hot
    ("knn", KNeighborsClassifier(
        n_neighbors=11, weights="distance", metric="manhattan"
    )),
])
knn.fit(Xtr, y_train)
y_pred_knn = knn.predict(Xte)

print("\n=== KNN — classification_report ===")
print(classification_report(y_test, y_pred_knn, digits=3, zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_knn, normalize="true", values_format=".2f"
)
plt.title("KNN — Confusion Matrix (normalized=true)")
plt.tight_layout()
plt.savefig(os.path.join("outputs", "KNN_confusion_matrix.png"), dpi=200)
plt.close()
