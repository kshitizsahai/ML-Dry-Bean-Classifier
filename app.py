import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree":       "decision_tree.joblib",
    "KNN":                 "knn.joblib",
    "Naive Bayes":         "naive_bayes.joblib",
    "SVM":                 "svm.joblib",
}


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(BASE, "model", MODEL_FILES[name]))


def compute_metrics(model, X, y):
    pred = model.predict(X)
    out = {
        "Accuracy":  accuracy_score(y, pred),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y, pred, average="weighted", zero_division=0),
        "F1":        f1_score(y, pred, average="weighted", zero_division=0),
        "MCC":       matthews_corrcoef(y, pred),
    }
    try:
        proba = model.predict_proba(X)
        out["AUC"] = roc_auc_score(y, proba, multi_class="ovr",
                                   average="weighted", labels=model.classes_)
    except ValueError:
        out["AUC"] = np.nan   # happens if the uploaded file lacks some classes
    return out, pred


# ---------------- sidebar ----------------
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + Class column)", type="csv")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
show_all = st.sidebar.checkbox("Also compare all 5 models", value=True)

# ---------------- main page ----------------
st.title("Dry Bean Variety Classifier")
st.caption("Five pre-trained scikit-learn models evaluated on your uploaded test data. "
           "Nothing is retrained here: the saved models are loaded and used for prediction only.")

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin. Use `test_data.csv` from the repo.")
    st.stop()

data = pd.read_csv(uploaded)
if TARGET not in data.columns:
    st.error(f"The file must contain the target column '{TARGET}'.")
    st.stop()

model = load_model(model_name)
needed = list(model.feature_names_in_)
missing = [c for c in needed if c not in data.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

X = data[needed]
y = data[TARGET]

c1, c2, c3 = st.columns(3)
c1.metric("Rows uploaded", len(data))
c2.metric("Features used", len(needed))
c3.metric("Classes in file", y.nunique())

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20))

# ---- selected model ----
metrics, pred = compute_metrics(model, X, y)
st.subheader(f"Evaluation metrics: {model_name}")
cols = st.columns(6)
for col, key in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    val = metrics[key]
    col.metric(key, "n/a" if pd.isna(val) else f"{val:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3))

# ---- all-model comparison ----
if show_all:
    st.subheader("All models on the uploaded data")
    rows = []
    for name in MODEL_FILES:
        m, _ = compute_metrics(load_model(name), X, y)
        rows.append({"Model": name, **m})
    comp = pd.DataFrame(rows).set_index("Model")[
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].round(4)
    st.dataframe(comp)

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    comp[["Accuracy", "F1", "MCC"]].plot(kind="bar", ax=ax2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Score")
    plt.xticks(rotation=20)
    fig2.tight_layout()
    st.pyplot(fig2)