import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay)

DATA_FILE = "Dry_Bean_Dataset.csv"   # or "Dry_Bean_Dataset.xlsx"
TARGET = "Class"
SEED = 42

# ---------- 1. load data ----------
if DATA_FILE.endswith(".xlsx"):
    df = pd.read_excel(DATA_FILE)
else:
    df = pd.read_csv(DATA_FILE)

print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print(df[TARGET].value_counts())

# basic cleaning
df = df.drop_duplicates().dropna().reset_index(drop=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ---------- 2. split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# test file for the Streamlit app (features + target)
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv with", len(test_df), "rows")

# ---------- 3. models (scaler lives inside each pipeline) ----------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000, random_state=SEED)),
    "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=SEED),
    "KNN":                 make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes":         GaussianNB(),
    "SVM":                 make_pipeline(StandardScaler(),
                                         SVC(kernel="rbf", probability=True, random_state=SEED)),
}

os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ---------- 4. train, evaluate, save ----------
results = []
for name, clf in models.items():
    print(f"Training {name} ...")
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    results.append({
        "ML Model Name": name,
        "Accuracy":  accuracy_score(y_test, pred),
        "AUC":       roc_auc_score(y_test, proba, multi_class="ovr",
                                   average="weighted", labels=clf.classes_),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1":        f1_score(y_test, pred, average="weighted", zero_division=0),
        "MCC":       matthews_corrcoef(y_test, pred),
    })

    fname = name.lower().replace(" ", "_")
    joblib.dump(clf, f"model/{fname}.joblib")

    # confusion matrix plot (matplotlib only)
    cm = confusion_matrix(y_test, pred, labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(f"plots/cm_{fname}.png", dpi=120)
    plt.close(fig)

results_df = pd.DataFrame(results).set_index("ML Model Name").round(4)
results_df.to_csv("model_comparison.csv")
print("\n=== Comparison table (paste into README) ===")
print(results_df)
print("\nMarkdown version:\n")
print(results_df.reset_index().to_markdown(index=False))