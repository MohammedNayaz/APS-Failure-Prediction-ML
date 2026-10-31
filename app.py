import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="APS Failure Prediction",
    page_icon="🚛",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

# --------------------------------------------------
# 2. Custom styling
# --------------------------------------------------

st.markdown("""
<style>

/* Blue outline when dropdown is focused */
div[data-baseweb="select"] > div:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 1px #2563EB !important;
}

/* Blue outline on hover */
div[data-baseweb="select"] > div:hover {
    border-color: #2563EB !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 2. Load the saved models
# --------------------------------------------------
@st.cache_resource
def load_models():
    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Saved model not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


# Read the expected input columns from our saved test dataset.
@st.cache_data
def load_feature_columns():
    reference_data = pd.read_csv(TEST_DATA_PATH)
    return [column for column in reference_data.columns
            if column != "class"]


# --------------------------------------------------
# 3. Page heading and instructions
# --------------------------------------------------
st.title("🚛 APS Failure Prediction")
st.subheader("Machine Learning Classifier Comparison")

st.write(
    "This application evaluates saved machine learning models "
    "on uploaded data from the APS Failure at Scania Trucks dataset. "
    "It does not retrain the models."
)

st.info(
    "Upload a CSV containing the same input features used during "
    "training and a target column named 'class'. "
    "The target labels should be 'neg' or 'pos'."
)

# --------------------------------------------------
# 4. Upload data and select a model
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"]
)

selected_model_name = st.selectbox(
    "Select a classifier",
    list(MODEL_FILES.keys())
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.write("### Uploaded Dataset Preview")
        st.dataframe(data.head())

        if "class" not in data.columns:
            st.error(
                "The uploaded CSV must contain a target column named 'class'."
            )
            st.stop()

        if data.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        # Load expected columns and saved models.
        feature_columns = load_feature_columns()
        models = load_models()

        # Check that all training features are available.
        missing_columns = [
            column for column in feature_columns
            if column not in data.columns
        ]

        if missing_columns:
            st.error(
                "Some required feature columns are missing: "
                + ", ".join(missing_columns[:20])
            )
            st.stop()

        # Keep the original feature order used during training.
        X_uploaded = data[feature_columns].copy()

        # Convert target labels into the same numeric format used in training.
        target = data["class"].astype(str).str.strip().str.lower()
        y_mapping = {"neg": 0, "pos": 1, "0": 0, "1": 1}
        y_true = target.map(y_mapping)

        if y_true.isna().any():
            st.error(
                "The class column must contain only 'neg', 'pos', "
                "0, or 1."
            )
            st.stop()

        y_true = y_true.astype(int)

        if y_true.nunique() < 2:
            st.warning(
                "The uploaded data contains only one target class. "
                "AUC may be unavailable."
            )

        # --------------------------------------------------
        # 5. Evaluate the selected saved model
        # --------------------------------------------------
        model = models[selected_model_name]
        y_pred = model.predict(X_uploaded)

        # Get scores needed for AUC.
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_uploaded)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_uploaded)
        else:
            y_score = None

        if y_score is not None and y_true.nunique() == 2:
            auc_value = roc_auc_score(y_true, y_score)
        else:
            auc_value = np.nan

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )
        mcc = matthews_corrcoef(y_true, y_pred)

        # --------------------------------------------------
        # 6. Display the evaluation metrics
        # --------------------------------------------------
        st.divider()
        st.header(f"Evaluation Results: {selected_model_name}")

        metric_columns = st.columns(3)

        metric_columns[0].metric("Accuracy", f"{accuracy:.4f}")
        metric_columns[1].metric(
            "AUC", f"{auc_value:.4f}" if not np.isnan(auc_value)
            else "N/A"
        )
        metric_columns[2].metric("Precision", f"{precision:.4f}")

        metric_columns = st.columns(3)

        metric_columns[0].metric("Recall", f"{recall:.4f}")
        metric_columns[1].metric("F1 Score", f"{f1:.4f}")
        metric_columns[2].metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # 7. Confusion matrix
        # --------------------------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual neg", "Actual pos"],
            columns=["Predicted neg", "Predicted pos"]
        )

        st.dataframe(cm_df)

        # --------------------------------------------------
        # 8. Classification report
        # --------------------------------------------------
        st.subheader("Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["neg", "pos"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4))

        # --------------------------------------------------
        # 9. Download evaluation results
        # --------------------------------------------------
        evaluation_results = pd.DataFrame([{
            "Model": selected_model_name,
            "Accuracy": accuracy,
            "AUC": auc_value,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc
        }])

        st.download_button(
            "Download Evaluation Metrics",
            data=evaluation_results.to_csv(index=False),
            file_name="evaluation_metrics.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Unable to evaluate the uploaded data: {error}")

else:
    st.write(
        "Upload a labeled CSV to display the evaluation results."
    )

st.divider()
st.caption(
    "Dev - Mohammed Nayaz | M.Tech AIML | Machine Learning Assignment 2 | "
    "APS Failure at Scania Trucks"
)
