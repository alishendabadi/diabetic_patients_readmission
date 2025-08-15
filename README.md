# Diabetic Patient Readmission Classification

This project performs classification of diabetic patients based on their readmission status:
- **NO**: No readmission
- **<30**: Readmitted within 30 days  
- **>30**: Readmitted after 30 days

The primary focus is on maximizing **Recall for the <30 days class**, as this is the most critical for early intervention.

## Features

- **Data Loading & Exploration**: Comprehensive data analysis and missing value handling
- **Preprocessing**: Standard normalization and categorical encoding
- **Class Imbalance Handling**: Three approaches - No balancing, Undersampling, and SMOTE
- **Multiple Models**: Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, MLP
- **Comprehensive Evaluation**: All common metrics with focus on <30 days recall
- **Visualizations**: Confusion matrices, feature importance, ROC curves, precision-recall curves, balancing comparison plots
- **Model Persistence**: Automatic saving of all trained models as pickle files
- **Explainability Analysis**: Comprehensive SHAP and LIME analysis for model interpretability

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your data file `diabetic2.csv` is in the same directory
2. Run the classification script:
```bash
python diabetic_readmission_classification.py
```

3. (Optional) Run the explainability analysis on the best model or any model you want:
```bash
python model_explainability.py
```

## Output

The script will generate:

### Console Output:
- Dataset overview and statistics
- Missing value analysis
- Model training progress
- Performance comparison table
- Detailed classification reports
- Best model identification for <30 days recall

## Key Metrics

The script evaluates models using:
- **Accuracy**: Overall prediction accuracy
- **Precision**: Precision for each class
- **Recall**: Recall for each class (especially <30 days)
- **F1-Score**: Harmonic mean of precision and recall
- **Recall_<30**: Special focus on recall for <30 days class

## Model Details

1. **Random Forest**: Ensemble of decision trees
2. **XGBoost**: Gradient boosting with regularization
3. **LightGBM**: Light Gradient Boosting Machine (fast gradient boosting)
4. **Logistic Regression**: Linear model with regularization
5. **SVM**: Support Vector Machine with RBF kernel
6. **MLP**: Multi-layer perceptron neural network

## Data Preprocessing

- **Missing Values**: Categorical variables filled with 'Unknown', numerical with median
- **Encoding**: Label encoding for categorical variables
- **Scaling**: StandardScaler for feature normalization
- **Train/Test Split**: 80/20 split with stratification
- **Class Balancing**: Three methods tested:
  - **No Balancing**: Original imbalanced dataset
  - **Undersampling**: Random undersampling to minority class size
  - **SMOTE**: Synthetic Minority Over-sampling Technique

## Focus on <30 Days Recall

The script specifically identifies and reports the best model for predicting <30 days readmissions, as this is crucial for:
- Early intervention strategies
- Resource allocation
- Patient care optimization
- Cost reduction

## Class Imbalance Handling

## Model Persistence

All trained models are automatically saved as pickle files in the `models/` directory. Each model file contains:
- The trained model
- The scaler used for feature normalization
- The target encoder for label decoding
- Model metadata (name, balance method, metrics, timestamp)

This allows you to:
- Load and use models later without retraining
- Deploy models in production
- Perform explainability analysis on saved models
- Compare models across different training sessions

## Explainability Analysis

The `model_explainability.py` script provides comprehensive model interpretability using:

### SHAP (SHapley Additive exPlanations)
- **Feature importance plots**: Bar charts of feature contributions

## Customization

You can modify the scripts to:
- Add more models
- Change hyperparameters
- Adjust evaluation metrics
- Modify visualization styles
- Add cross-validation
- Implement ensemble methods
- Customize explainability methods
- Add more interpretability techniques

## Troubleshooting

If you encounter issues:
1. Ensure all required packages are installed
2. Check that `diabetic2.csv` is in the correct location
3. Verify the CSV file format and column names
4. Check Python version compatibility (3.7+ recommended) 