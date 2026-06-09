# Multi-Class Patient Outcome Prediction for Cirrhosis

Multi-class classification pipeline predicting patient outcomes (C / CL / D) using a stacking ensemble with strict leakage prevention.

## Pipeline Overview
- **Preprocessing**: ColumnTransformer with separate numerical (mean imputation + StandardScaler) and categorical (most-frequent imputation + OneHotEncoder) pipelines
- **Base models**: Logistic Regression (multinomial, C=0.1) + XGBoost (multi:softprob, 200 estimators, lr=0.2)
- **Meta-learner**: Logistic Regression trained on out-of-fold predictions
- **Validation**: 5-fold stratified CV; metric = multi-class log loss
- **Inference**: Serialisable pipeline — consistent transforms from training through test, ready for REST API integration

## Folder Structure
cirrhosis-outcome-prediction/
├── README.md
├── notebook/
│   └── Multi-Class_Patient_Outcome_Prediction_for_Cirrhosis.ipynb
├── outputs/
│   └── submission.csv
└── requirements.txt

## Tech Stack
Python · XGBoost · Scikit-Learn · Pandas · NumPy

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook/Multi-Class_Patient_Outcome_Prediction_for_Cirrhosis.ipynb
```

## Key Design Decisions
- StackingClassifier uses CV=5 for generating meta-features → prevents overfitting in the second layer
- Full pipeline object is fit-once and predict-anywhere — no manual preprocessing on test data
- Stratified split maintains class distribution across folds
