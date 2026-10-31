
# APS Failure Prediction Using Machine Learning

## 1. Project Overview

This project compares five machine learning classifiers for predicting failures in the Air Pressure System (APS) of heavy-duty trucks. The objective is to evaluate how different classification algorithms identify positive failure cases and negative cases using vehicle sensor measurements.

The project is developed as part of the M.Tech AIML Machine Learning Assignment 2.

## 2. Problem Statement

Unexpected failures in a truck's Air Pressure System can lead to operational disruption and maintenance costs. A machine learning model can help identify vehicles that may experience an APS failure based on sensor data.

This project trains and evaluates five classification algorithms on the same dataset and compares their performance using six evaluation metrics.

## 3. Dataset Description

**Dataset:** APS Failure at Scania Trucks  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks

The dataset contains sensor measurements from heavy-duty trucks. The target variable is `class`, with:

- `neg`: APS failure did not occur.
- `pos`: APS failure occurred.

The dataset contains more than 12 input features and 500 instances. The experiments use the labeled training file and a stratified train-test split for model evaluation.

The dataset contains missing values and a strong imbalance between the negative and positive classes.

## 4. Data Preprocessing

The following preprocessing steps are used:

1. The target column `class` is separated from the input features.
2. Target labels are encoded as `neg = 0` and `pos = 1`.
3. The dataset is divided into training and testing subsets using a stratified split.
4. Missing numerical values are handled using median imputation.
5. Standardization is applied to models that are sensitive to feature scales.
6. The preprocessing operations are included in the saved model pipelines to keep prediction-time processing consistent with training.

## 5. Machine Learning Models

The following five classifiers are implemented:

1. **Logistic Regression** — a linear classification model. Class weighting is used to help account for class imbalance.
2. **Decision Tree** — a tree-based model that learns feature-based decision rules.
3. **K-Nearest Neighbors (KNN)** — predicts a class using the labels of nearby training examples.
4. **Gaussian Naive Bayes** — a probabilistic classifier based on conditional independence assumptions.
5. **Support Vector Machine (LinearSVC)** — a linear margin-based classifier. Class weighting is used to help account for class imbalance.

All five models are evaluated on the same held-out test subset.

## 6. Evaluation Metrics

The models are compared using six metrics:

- **Accuracy:** proportion of all predictions that are correct.
- **AUC:** ability to rank positive examples above negative examples across classification thresholds.
- **Precision:** proportion of predicted positive cases that are actually positive.
- **Recall:** proportion of actual positive cases correctly identified.
- **F1 Score:** harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** balanced measure of the quality of binary predictions, taking all four confusion-matrix outcomes into account.

Because positive APS failures are rare, accuracy alone is not sufficient to determine the best model. Recall, F1, AUC and MCC are also considered.

## 7. Model Comparison

The following table presents the measured performance of the five classifiers
on the held-out test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9715 | 0.9622 | 0.3586 | 0.9000 | 0.5128 | 0.5580 |
| Decision Tree | 0.9873 | 0.8043 | 0.6212 | 0.6150 | 0.6181 | 0.6117 |
| KNN | 0.9899 | 0.9263 | 0.7801 | 0.5500 | 0.6452 | 0.6503 |
| Naive Bayes | 0.9664 | 0.9674 | 0.3178 | 0.8850 | 0.4676 | 0.5189 |
| Support Vector Machine | 0.9718 | 0.9578 | 0.3590 | 0.8850 | 0.5108 | 0.5535 |

### Model-wise Observations

**Logistic Regression:** Achieves a recall of 0.9000, identifying a high
proportion of actual positive failure cases. However, its precision of
0.3586 indicates that many predicted failures are false positives.

**Decision Tree:** Achieves an accuracy of 0.9873 and an F1 score of 0.6181.
Its precision and recall are more balanced than those of the high-recall
linear models. Its AUC of 0.8043 is the lowest among the five classifiers.

**K-Nearest Neighbors (KNN):** Achieves the highest accuracy (0.9899),
precision (0.7801), F1 score (0.6452), and MCC (0.6503). It provides the
strongest balance of precision and recall according to F1, and the strongest
overall binary classification association according to MCC in these results.
Its recall of 0.5500 is a limitation because it misses a substantial fraction
of actual failures.

**Gaussian Naive Bayes:** Achieves the highest AUC of 0.9674 and a recall of
0.8850. However, its precision of 0.3178 and F1 score of 0.4676 are relatively
low, indicating many false-positive predictions.

**Support Vector Machine (LinearSVC):** Achieves a recall of 0.8850 and an
AUC of 0.9578. Like Logistic Regression, it detects many positive cases but
has relatively low precision (0.3590).

### Overall Conclusion

KNN is the strongest model by F1 score and MCC in this experiment, while
Naive Bayes achieves the highest AUC and Logistic Regression achieves the
highest recall. Therefore, the preferred classifier depends on the operational
objective.

For failure detection, recall is particularly important because missed failures
may be costly. However, a model with high recall may also generate more false
alarms. The results show that KNN offers a better precision-recall balance,
but its lower recall should be considered before selecting it for a real
maintenance application.

These results are based on the current train-test split and model settings.
They do not establish that one classifier will be best on every dataset or
operating condition.

## 8. Streamlit Application

The Streamlit application provides the following functionality:

- Upload a labeled CSV file.
- Select one of the five saved classifiers.
- Evaluate the selected model on the uploaded data.
- Display Accuracy, AUC, Precision, Recall, F1 and MCC.
- Display the confusion matrix and classification report.
- Download the evaluation metrics as a CSV file.

The application loads previously trained models and does not retrain them when a user uploads a test file.

## 9. Project Structure

```text
ML_Assignment_2/
├── app.py
├── README.md
├── requirements.txt
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── data/
│   └── test_data.csv
└── results/
    └── model_comparison.csv
```

## 10. Installation and Execution

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application from the project directory:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in the terminal.

## 11. Links

- **GitHub Repository:** To be added after repository creation.
- **Live Streamlit Application:** To be added after deployment.
- **Dataset:** https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks

## 12. Conclusion

This project provides a comparative evaluation of five machine learning classifiers for APS failure prediction. The final conclusion will identify the strongest model based on the measured results and discuss the trade-offs between identifying failure cases and avoiding false alarms.
