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
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn shap
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

3. (Optional) Run the explainability analysis on the best model:
```bash
python model_explainability.py
```

4. (Optional) Load and use a saved model for predictions:
```bash
python load_and_use_model.py
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
- `models/` - Directory containing all trained models as pickle files
- `model_performance_comparison.png` - Bar charts comparing model performance
- `confusion_matrices.png` - Confusion matrices for all models
- `feature_importance.png` - Feature importance plots for tree-based models
- `roc_curves.png` - ROC curves for all models and classes
- `precision_recall_curves.png` - Precision-recall curves focusing on <30 days class
- `balancing_recall_comparison.png` - Recall comparison across balancing methods
- `balancing_accuracy_comparison.png` - Accuracy comparison across balancing methods
- `balancing_f1_comparison.png` - F1-score comparison across balancing methods
- `balancing_metrics_heatmap.png` - Heatmap of all metrics across methods

### Explainability Analysis Files (when running model_explainability.py):
- `shap_summary_*.png` - SHAP summary plots
- `shap_importance_*.png` - SHAP feature importance plots
- `shap_waterfall_sample_*.png` - SHAP waterfall plots for individual predictions
- `shap_dependence_*.png` - SHAP dependence plots for top features
- `lime_explanation_sample_*.png` - LIME explanations for individual predictions
- `permutation_importance_*.png` - Permutation importance plots
- `feature_distributions_*.png` - Feature distribution plots
- `explainability_report_*.md` - Comprehensive explainability report

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

The script automatically tests three approaches to handle class imbalance:

1. **No Balancing**: Uses the original imbalanced dataset
2. **Undersampling**: Reduces majority classes to match minority class size
3. **SMOTE**: Creates synthetic samples for minority classes

This comprehensive comparison helps identify the best approach for maximizing recall on the critical <30 days class.

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
- **Summary plots**: Overall feature importance and effects
- **Feature importance plots**: Bar charts of feature contributions
- **Waterfall plots**: Individual prediction explanations
- **Dependence plots**: Feature interaction effects

### LIME (Local Interpretable Model-agnostic Explanations)
- **Local explanations**: Individual prediction interpretations
- **Feature contributions**: Which features contributed to each prediction

### Additional Methods
- **Permutation importance**: Model-agnostic feature importance
- **Feature distributions**: Understanding feature patterns
- **Comprehensive reports**: Detailed analysis summaries

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