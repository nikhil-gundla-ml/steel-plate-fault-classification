# Steel Plate Fault Classification Using Machine Learning

## 1. Project Overview

This project develops a machine learning application to classify steel plate surface faults using five classification algorithms. A Streamlit web application allows users to upload a test dataset, compare model performance and view individual model evaluation results.

## 2. Dataset

**Source:** UCI Machine Learning Repository  
**Dataset:** Steel Plates Faults  
**Dataset ID:** 198  
**Source URL:** https://archive.ics.uci.edu/dataset/198/steel+plates+faults

- 1,941 instances
- 27 numerical input features
- 7 fault categories
- Training and testing split: 80:20
- Random state: 42

The seven fault categories are Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps and Other_Faults.

## 3. Machine Learning Algorithms

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Evaluation Metrics

Each model is evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The baseline evaluation identified Support Vector Machine as the best-performing model based on weighted F1 score.

## 5. Project Structure

```text
steel-plate-fault-classification/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── steel_plates_full.csv
│   ├── steel_plates_raw.csv
│   ├── train_data.csv
│   └── test_data.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   ├── model_comparison.csv
│   ├── class_labels.json
│   └── feature_names.json
└── notebooks/
    ├── 01_dataset_inspection.ipynb
    ├── 02_model_training_evaluation.ipynb
    └── data/
        └── steel_plates_raw.csv
```

## 6. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/nikhil-gundla-ml/steel-plate-fault-classification.git
cd steel-plate-fault-classification
```

Create and activate a Python virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 7. Streamlit Application Features

- Upload a CSV file containing 27 input features and the Target column.
- Validate the uploaded dataset.
- Compare the performance of all five trained models.
- Select an individual model for detailed evaluation.
- View evaluation metrics, classification reports and confusion matrices.
- Load saved models without retraining on uploaded test data.

## 8. Model Results

The baseline Support Vector Machine model achieved:

- Accuracy: 75.06%
- Weighted F1 Score: 75.21%
- AUC: 89.74%
- MCC: 0.6769

Logistic Regression achieved the highest baseline AUC of approximately 90.66%.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7275 | 0.9066 | 0.7329 | 0.7275 | 0.7277 | 0.6487 |
| Decision Tree | 0.7429 | 0.8283 | 0.7455 | 0.7429 | 0.7429 | 0.6688 |
| KNN | 0.7275 | 0.8941 | 0.7318 | 0.7275 | 0.7245 | 0.6548 |
| Gaussian Naive Bayes | 0.4524 | 0.7955 | 0.5741 | 0.4524 | 0.4003 | 0.3789 |
| Support Vector Machine | **0.7506** | 0.8974 | 0.7578 | 0.7506 | **0.7521** | **0.6769** |

## 9. Academic Project

Developed as part of the Machine Learning coursework for the M.Tech Artificial Intelligence and Machine Learning programme at BITS Pilani Work Integrated Learning Programmes (WILP).

