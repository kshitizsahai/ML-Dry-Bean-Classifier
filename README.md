# ML Assignment 2: Dry Bean Classification

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP  
**Name:** Kshitiz Sahai  
**BITS ID:** 2026AJ05035  
**Date:** 08-11-2026  
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

### Comparison table (held-out test set, 2,709 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9195 | 0.9935 | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.9022 | 0.9634 | 0.9025 | 0.9022 | 0.9023 | 0.8817 |
| KNN | 0.9118 | 0.9832 | 0.9123 | 0.9118 | 0.9119 | 0.8933 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Support Vector Machine | 0.9210 | 0.9933 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

*Precision, Recall and F1 are support-weighted averages, and AUC is one-vs-rest
weighted. Weighted recall is mathematically identical to accuracy in
multi-class classification, which is why those two columns match.*
### Confusion matrices

Each matrix below shows the model's predictions on the 2,709-sample test set.
Rows are the true bean variety and columns are the predicted variety, so the
diagonal counts correct predictions and off-diagonal cells show which varieties
the model confuses.

All five models classify BOMBAY perfectly, and the main
errors are between DERMASON and SIRA. Naive Bayes shows the most spread outside
the diagonal, especially for BARBUNYA.

| Logistic Regression | Decision Tree |
|---|---|
| ![Logistic Regression](plots/cm_logistic_regression.png) | ![Decision Tree](plots/cm_decision_tree.png) |

| KNN | Naive Bayes |
|---|---|
| ![KNN](plots/cm_knn.png) | ![Naive Bayes](plots/cm_naive_bayes.png) |

| Support Vector Machine |
|---|
| ![SVM](plots/cm_svm.png) |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Reached 91.95% accuracy and the highest AUC (0.9935), meaning its class probabilities rank samples very well. Reaching this level with a linear boundary suggests that, after scaling, most bean varieties are close to linearly separable in this feature space. Its main errors were DERMASON samples predicted as SIRA (53) and SIRA predicted as DERMASON (41). It was also fast to train and easy to interpret. |
| Decision Tree | Scored 90.22% accuracy, the lowest of the four stronger models, and its AUC (0.9634) was well below the other non-Naive-Bayes models. Tree leaves give coarse probability estimates, which lowers AUC even when the predicted labels are mostly right. It made the most DERMASON/SIRA mistakes (61 and 57) and had the weakest SIRA recall (about 84%). Axis-aligned splits struggle where classes overlap along a diagonal combination of correlated features. |
| KNN | Scored 91.18% accuracy with AUC 0.9832, between the linear models and the tree. Feature scaling was essential here, because Area (tens of thousands) would otherwise dominate distances. It still confused DERMASON and SIRA (53 and 55 errors), which implies those two classes sit close together in feature space. It is slower at prediction time because it stores and searches the full training set. |
| Naive Bayes | Clearly the weakest, with 76.30% accuracy and MCC 0.714, about 15 points of accuracy below the best model. Many features are strongly correlated (Area, Perimeter, ConvexArea, EquivDiameter and the axis lengths), which violates the independence assumption, and the Gaussian assumption is a poor fit for skewed bean measurements. BARBUNYA suffered most: only about 46% were recalled, with 108 predicted as CALI. Its AUC (0.9644) stayed high despite low accuracy, so it ranks samples reasonably but its hard class decisions are poor. |
| Support Vector Machine | Best on five of six metrics: accuracy 92.10%, F1 0.9211 and MCC 0.9045. The RBF kernel can bend the decision boundary where classes overlap, giving the best recall for CALI (about 95%), SEKER (about 95%) and DERMASON (about 91%). Its remaining errors are still the DERMASON/SIRA pair (49 and 47), the smallest counts among the models except Naive Bayes. The costs are a longer training time and the need for scaling. |
| **Overall winner for your dataset** | **Support Vector Machine.** It has the highest accuracy, F1 and MCC, and its AUC (0.9933) is essentially tied with Logistic Regression (0.9935). Its lead over Logistic Regression is only 4 correctly classified test samples (2,495 vs 2,491), so the two are practically equivalent on this split. SVM is chosen as the winner for the best overall balance across all metrics, but Logistic Regression is a very close and simpler alternative. |

### Additional observations
- **BOMBAY** was classified perfectly (104 of 104) by all five models. Bombay beans are much larger than the other varieties, so size features separate them easily.
- **DERMASON vs SIRA** is the dominant error pair for every model, which points to genuine overlap between these two varieties rather than a weakness of one algorithm.
- **BARBUNYA vs CALI** is the second most common confusion (for example 18 and 7 for SVM, but 108 and 42 for Naive Bayes).
- Because the classes are imbalanced (BOMBAY is under 4% of the data), MCC is a useful extra check. It tracks the same ranking as accuracy and F1 here, which confirms the ordering is not an artifact of the majority classes.

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
