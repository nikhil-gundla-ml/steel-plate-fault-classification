
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Steel Plate Fault Classification",
    page_icon="🏭",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"


# --------------------------------------------------
# 2. LOAD SAVED MODELS AND METADATA
# --------------------------------------------------

@st.cache_resource
def load_models():

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "support_vector_machine.joblib"
    }

    loaded = {}

    for name, filename in model_files.items():
        loaded[name] = joblib.load(MODEL_DIR / filename)

    return loaded


@st.cache_data
def load_metadata():

    with open(MODEL_DIR / "feature_names.json") as file:
        features = json.load(file)

    with open(MODEL_DIR / "class_labels.json") as file:
        labels = json.load(file)

    return features, labels


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("🏭 Steel Plate Fault Classification")

st.write(
    "A machine learning application to classify "
    "steel plate surface defects using five trained "
    "classification algorithms."
)

st.divider()

st.subheader("Project Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Input Features", "27")
col2.metric("Fault Categories", "7")
col3.metric("ML Models", "5")


# --------------------------------------------------
# 4. LOAD MODEL FILES
# --------------------------------------------------

try:
    models = load_models()
    feature_names, class_names = load_metadata()

except Exception as error:
    st.error(f"Unable to load saved model files: {error}")
    st.stop()


# --------------------------------------------------
# 5. TEST DATA UPLOAD
# --------------------------------------------------

st.header("1. Upload Test Dataset")

st.info(
    "Upload a CSV file containing all 27 input features "
    "and the Target column. The application will evaluate "
    "the saved models without retraining them."
)

uploaded_file = st.file_uploader(
    "Choose your test CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the test_data.csv file to continue.")
    st.stop()


# --------------------------------------------------
# 6. VALIDATE UPLOADED DATA
# --------------------------------------------------

try:
    test_data = pd.read_csv(uploaded_file)

    required_columns = feature_names + ["Target"]

    missing_columns = [
        col for col in required_columns
        if col not in test_data.columns
    ]

    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    test_data = test_data[required_columns].copy()

    if test_data.empty:
        st.error("The uploaded CSV contains no records.")
        st.stop()

    if test_data.isnull().sum().sum() > 0:
        st.error("The uploaded CSV contains missing values.")
        st.stop()

    if not all(
        pd.api.types.is_numeric_dtype(test_data[col])
        for col in feature_names
    ):
        st.error("All 27 input features must be numerical.")
        st.stop()

    if not test_data["Target"].isin(class_names).all():
        st.error("The Target column contains unknown fault categories.")
        st.stop()

    if test_data["Target"].nunique() < 2:
        st.error("The uploaded data must contain at least two classes for evaluation.")
        st.stop()

except Exception as error:
    st.error(f"Unable to read uploaded CSV: {error}")
    st.stop()


X_test = test_data[feature_names]
y_test = test_data["Target"]


st.success("Test dataset uploaded and validated successfully.")

st.write("Uploaded dataset dimensions:", test_data.shape)

with st.expander("Preview uploaded dataset"):
    st.dataframe(test_data.head(10), width="stretch")


# --------------------------------------------------
# 7. EVALUATE ALL FIVE MODELS
# --------------------------------------------------

st.header("2. Model Performance Comparison")

results = {}
predictions = {}
probabilities = {}

for model_name, model in models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    model_classes = model.classes_

    auc = roc_auc_score(
        y_test,
        y_prob,
        labels=model_classes,
        multi_class="ovr",
        average="weighted"
    )

    predictions[model_name] = y_pred
    probabilities[model_name] = y_prob

    results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }


comparison_df = pd.DataFrame.from_dict(
    results,
    orient="index"
).reset_index(names="Model")

comparison_df = comparison_df.sort_values(
    by="F1 Score",
    ascending=False
)

st.dataframe(
    comparison_df.style.format({
        col: "{:.4f}"
        for col in comparison_df.columns
        if col != "Model"
    }),
    use_container_width=True
)


# --------------------------------------------------
# 8. SELECT A MODEL
# --------------------------------------------------

st.header("3. Individual Model Evaluation")

selected_model = st.selectbox(
    "Select a classification model",
    list(models.keys())
)

selected_predictions = predictions[selected_model]


# --------------------------------------------------
# 9. DISPLAY SELECTED MODEL METRICS
# --------------------------------------------------

selected_metrics = results[selected_model]

metric_cols = st.columns(3)

metric_cols[0].metric(
    "Accuracy",
    f"{selected_metrics['Accuracy']:.4f}"
)

metric_cols[1].metric(
    "AUC Score",
    f"{selected_metrics['AUC']:.4f}"
)

metric_cols[2].metric(
    "MCC Score",
    f"{selected_metrics['MCC']:.4f}"
)

metric_cols = st.columns(3)

metric_cols[0].metric(
    "Precision",
    f"{selected_metrics['Precision']:.4f}"
)

metric_cols[1].metric(
    "Recall",
    f"{selected_metrics['Recall']:.4f}"
)

metric_cols[2].metric(
    "F1 Score",
    f"{selected_metrics['F1 Score']:.4f}"
)


# --------------------------------------------------
# 10. CLASSIFICATION REPORT
# --------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_test,
    selected_predictions,
    labels=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.4f}"),
    use_container_width=True
)


# --------------------------------------------------
# 11. CONFUSION MATRIX
# --------------------------------------------------

st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(9, 7))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    selected_predictions,
    labels=class_names,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=False,
    ax=ax
)

ax.set_title(f"Confusion Matrix - {selected_model}")

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 12. FINAL SUMMARY
# --------------------------------------------------

st.divider()

st.subheader("Evaluation Summary")

best_model = comparison_df.iloc[0]["Model"]

st.write(
    f"Based on weighted F1 score, "
    f"the best-performing model on the uploaded dataset is: "
    f"**{best_model}**."
)

st.caption(
    "All models were loaded from saved files. "
    "No model training was performed on the uploaded test data."
)
