# Dry Bean Classification: ML Assignment 2

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP  
**Name / ID:** Kshitiz Sahai / 2026AJ05035  
---

## a. Problem Statement

Classify dry bean grains into one of seven varieties from
shape and size measurements extracted from images. Five classification models
are trained on the same dataset, compared using six metrics, and deployed as an
interactive Streamlit app.

---

## b. Dataset Description

| Item | Details |
|---|---|
| Source | UCI Machine Learning Repository, Dry Bean Dataset (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| Citation | Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B |
| License | CC BY 4.0 |
| Task | Multi-class classification (7 classes) |
| Instances | 13,611 originally; 13,543 after removing 68 duplicate rows |
| Features | 16, all numeric (2 integer, 14 continuous) |
| Target | `Class` (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA) |
| Missing values | None |
| Duplicates | 68 exact duplicate rows (all in HOROZ), removed before splitting |
| Train/test split | 80/20, stratified, random_state=42 (about 10,834 train / 2,709 test rows) |
| Preprocessing | Duplicates removed; StandardScaler applied inside the pipeline for Logistic Regression, KNN and SVM; Decision Tree and Naive Bayes use raw features |

**Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness,
Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

The features are shape and size measurements taken from images of individual
bean grains. Their scales differ greatly (Area ranges from about 20,420 to
254,616, while Solidity stays between about 0.92 and 0.995), so scaling is
needed for distance-based and margin-based models.

**Class distribution (after removing duplicates):**

| Class | Original count | Count used | Share |
|---|---|---|---|
| DERMASON | 3,546 | 3,546 | 26.18% |
| SIRA | 2,636 | 2,636 | 19.46% |
| SEKER | 2,027 | 2,027 | 14.97% |
| HOROZ | 1,928 | 1,860 | 13.73% |
| CALI | 1,630 | 1,630 | 12.04% |
| BARBUNYA | 1,322 | 1,322 | 9.76% |
| BOMBAY | 522 | 522 | 3.85% |
| **Total** | **13,611** | **13,543** | **100%** |  

The data is moderately imbalanced: the largest class (DERMASON) has about 6.8
times as many rows as the smallest (BOMBAY). Stratified splitting keeps these
proportions the same in the train and test sets, and weighted averaging is used
for precision, recall and F1.

---

## c. GitHub Repository Link

https://github.com/kshitizsahai/ML-Dry-Bean-Classifier.git

**Live Streamlit app:** https://ml-dry-bean-classifier-v1.streamlit.app/

**Repository structure:**
```
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

---

## d. Models Used

### Comparison table (evaluated on the held-out test set)

| ML Model Name       |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.9195 | 0.9935 |      0.9201 |   0.9195 | 0.9197 | 0.9028 |
| Decision Tree       |     0.9022 | 0.9634 |      0.9025 |   0.9022 | 0.9023 | 0.8817 |
| KNN                 |     0.9118 | 0.9832 |      0.9123 |   0.9118 | 0.9119 | 0.8933 |
| Naive Bayes         |     0.763  | 0.9644 |      0.7647 |   0.763  | 0.7607 | 0.7143 |
| SVM                 |     0.921  | 0.9933 |      0.9214 |   0.921  | 0.9211 | 0.9045 |

*Precision, Recall and F1 are weighted averages. AUC is one-vs-rest, weighted.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <your analysis> |
| Decision Tree | <your analysis> |
| KNN | <your analysis> |
| Naive Bayes | <your analysis> |
| Support Vector Machine | <your analysis> |
| **Overall winner for your dataset** | <model name, plus 2-3 sentences of justification> |

---

## Streamlit App Features

- CSV upload for test data
- Model selection dropdown (5 models)
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- Comparison of all models on the uploaded data

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
