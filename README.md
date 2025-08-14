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
- **Multiple Models**: Random Forest, XGBoost, Logistic Regression, SVM, MLP
- **Comprehensive Evaluation**: All common metrics with focus on <30 days recall
- **Visualizations**: Confusion matrices, feature importance, ROC curves, precision-recall curves, balancing comparison plots

## Installation

1. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your data file `diabetic2.csv` is in the same directory
2. Run the classification script:
```bash
python diabetic_readmission_classification.py
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

### Files Generated:
- `model_comparison_results.csv` - Detailed performance metrics for all balancing methods
- `model_performance_comparison.png` - Bar charts comparing model performance
- `confusion_matrices.png` - Confusion matrices for all models
- `feature_importance.png` - Feature importance plots for tree-based models
- `roc_curves.png` - ROC curves for all models and classes
- `precision_recall_curves.png` - Precision-recall curves focusing on <30 days class
- `balancing_recall_comparison.png` - Recall comparison across balancing methods
- `balancing_accuracy_comparison.png` - Accuracy comparison across balancing methods
- `balancing_f1_comparison.png` - F1-score comparison across balancing methods
- `balancing_metrics_heatmap.png` - Heatmap of all metrics across methods

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
3. **Logistic Regression**: Linear model with regularization
4. **SVM**: Support Vector Machine with RBF kernel
5. **MLP**: Multi-layer perceptron neural network

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

The script automatically tests three approaches to handle class imbalance:

1. **No Balancing**: Uses the original imbalanced dataset
2. **Undersampling**: Reduces majority classes to match minority class size
3. **SMOTE**: Creates synthetic samples for minority classes

This comprehensive comparison helps identify the best approach for maximizing recall on the critical <30 days class.

## Customization

You can modify the script to:
- Add more models
- Change hyperparameters
- Adjust evaluation metrics
- Modify visualization styles
- Add cross-validation
- Implement ensemble methods

## Troubleshooting

If you encounter issues:
1. Ensure all required packages are installed
2. Check that `diabetic2.csv` is in the correct location
3. Verify the CSV file format and column names
4. Check Python version compatibility (3.7+ recommended) 