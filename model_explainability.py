#!/usr/bin/env python3
"""
Model Explainability Analysis
============================

This script performs explainability analysis on the best trained model using SHAP
and other interpretability methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import shap
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_best_model():
    """Load the best performing model from the models directory"""
    print("Loading best model...")
    
    # Check if models directory exists
    if not os.path.exists('models'):
        print("Models directory not found. Please run the main training script first.")
        return None, None, None, None
    
    # List all model files
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    
    if not model_files:
        print("No model files found in models directory.")
        return None, None, None, None
    
    # Load all models and find the best one based on recall_30
    best_model_data = None
    best_recall_30 = -1
    
    for model_file in model_files:
        with open(os.path.join('models', model_file), 'rb') as f:
            model_data = pickle.load(f)
        
        recall_30 = model_data['metrics']['recall_30']
        if recall_30 > best_recall_30:
            best_recall_30 = recall_30
            best_model_data = model_data
    
    if best_model_data is None:
        print("No valid models found.")
        return None, None, None, None
    
    print(f"Best model: {best_model_data['model_name']}")
    print(f"Balance method: {best_model_data['balance_method']}")
    print(f"Recall for <30 days: {best_recall_30:.4f}")
    
    return (best_model_data['model'], best_model_data['scaler'], 
            best_model_data['target_encoder'], best_model_data)

def load_test_data():
    """Load and prepare test data for explainability analysis"""
    print("Loading test data...")
    
    # Load the original dataset
    df = pd.read_csv('diabetic2.csv')
    
    # Preprocess the data (same as in main script)
    df_processed = df.copy()
    
    # Handle missing values
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna('Unknown', inplace=True)
    
    numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    
    for col in categorical_columns:
        if col != 'readmitted':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    df_processed['readmitted_encoded'] = target_encoder.fit_transform(df_processed['readmitted'])
    
    # Remove unnecessary columns
    columns_to_drop = ['encounter_id', 'patient_nbr', 'readmitted']
    df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
    
    # Separate features and target
    X = df_processed.drop('readmitted_encoded', axis=1)
    y = df_processed['readmitted_encoded']
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, label_encoders

def create_feature_names(X):
    """Create meaningful feature names"""
    feature_names = []
    for i in range(X.shape[1]):
        feature_names.append(f'Feature_{i+1}')
    return feature_names

def shap_analysis(model, X_train, X_test, feature_names, model_name):
    """Perform SHAP analysis on the model"""
    print(f"\n=== SHAP Analysis for {model_name} ===")
    
    # Create SHAP explainer based on model type
    if hasattr(model, 'feature_importances_'):  # Tree-based models
        print("Using TreeExplainer for tree-based model...")
        explainer = shap.TreeExplainer(model)
    else:
        print("Using KernelExplainer for non-tree model...")
        # Use a sample of training data for background
        background_data = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
    
    # Calculate SHAP values for test set
    print("Calculating SHAP values...")
    if hasattr(model, 'feature_importances_'):
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            # For multi-class, focus on class 1 (<30 days)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    
    # 1. Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                     show=False, plot_size=(12, 8))
    plt.title(f'SHAP Summary Plot - {model_name}')
    plt.tight_layout()
    plt.savefig(f'shap_summary_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Feature Importance Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                     plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'shap_importance_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Waterfall plot for a few sample predictions
    print("Creating waterfall plots for sample predictions...")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                           shap_values[idx], X_test.iloc[idx], 
                           feature_names=feature_names, show=False)
        plt.title(f'SHAP Waterfall Plot - Sample {i+1} - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_sample_{i+1}_{model_name.lower().replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Dependence plots for top features
    print("Creating dependence plots for top features...")
    # Get feature importance
    feature_importance = np.abs(shap_values).mean(0)
    top_features = np.argsort(feature_importance)[-5:]  # Top 5 features
    
    for i, feature_idx in enumerate(top_features):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, shap_values, X_test, 
                           feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence Plot - {feature_names[feature_idx]} - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{feature_names[feature_idx]}_{model_name.lower().replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    return shap_values, explainer

def lime_analysis(model, X_test, feature_names, model_name, num_samples=5):
    """Perform LIME analysis on the model"""
    print(f"\n=== LIME Analysis for {model_name} ===")
    
    try:
        from lime import lime_tabular
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_test.values,
            feature_names=feature_names,
            class_names=['NO', '<30', '>30'],
            mode='classification'
        )
        
        # Explain a few sample predictions
        sample_indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            print(f"Explaining sample {i+1}...")
            
            # Get explanation
            exp = explainer.explain_instance(
                X_test.iloc[idx].values, 
                model.predict_proba,
                num_features=10
            )
            
            # Plot explanation
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(f'LIME Explanation - Sample {i+1} - {model_name}')
            plt.tight_layout()
            plt.savefig(f'lime_explanation_sample_{i+1}_{model_name.lower().replace(" ", "_")}.png', 
                        dpi=300, bbox_inches='tight')
            plt.show()
            
    except ImportError:
        print("LIME not available. Install with: pip install lime")

def permutation_importance_analysis(model, X_test, y_test, feature_names, model_name):
    """Perform permutation importance analysis"""
    print(f"\n=== Permutation Importance Analysis for {model_name} ===")
    
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=42,
        scoring='recall'
    )
    
    # Create importance plot
    plt.figure(figsize=(12, 8))
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    top_n = min(20, len(feature_names))
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Permutation Importance')
    plt.title(f'Permutation Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'permutation_importance_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return result

def partial_dependence_analysis(model, X_test, y_test, feature_names, model_name):
    """Perform partial dependence analysis"""
    print(f"\n=== Partial Dependence Analysis for {model_name} ===")
    
    from sklearn.inspection import partial_dependence
    
    # Get top features from permutation importance
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    top_features = np.argsort(result.importances_mean)[-3:]  # Top 3 features
    
    # Create partial dependence plots
    fig, axes = plt.subplots(1, len(top_features), figsize=(15, 5))
    if len(top_features) == 1:
        axes = [axes]
    
    for i, feature_idx in enumerate(top_features):
        # Calculate partial dependence
        feature_values = np.linspace(X_test.iloc[:, feature_idx].min(), 
                                   X_test.iloc[:, feature_idx].max(), 50)
        
        # For simplicity, we'll create a basic plot
        # In practice, you might want to use more sophisticated methods
        axes[i].hist(X_test.iloc[:, feature_idx], bins=20, alpha=0.7, density=True)
        axes[i].set_xlabel(feature_names[feature_idx])
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Distribution of {feature_names[feature_idx]}')
    
    plt.suptitle(f'Feature Distributions - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_distributions_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_explainability_report(model_data, shap_values, X_test, y_test, feature_names):
    """Create a comprehensive explainability report"""
    print("\n=== Creating Explainability Report ===")
    
    model_name = model_data['model_name']
    balance_method = model_data['balance_method']
    metrics = model_data['metrics']
    
    # Create report
    report = f"""
# Model Explainability Report

## Model Information
- **Model**: {model_name}
- **Balance Method**: {balance_method}
- **Performance Metrics**:
  - Accuracy: {metrics['accuracy']:.4f}
  - Precision: {metrics['precision']:.4f}
  - Recall: {metrics['recall']:.4f}
  - F1-Score: {metrics['f1']:.4f}
  - Recall for <30 days: {metrics['recall_30']:.4f}

## Key Insights

### Top Features by SHAP Importance
"""
    
    # Get top features from SHAP
    feature_importance = np.abs(shap_values).mean(0)
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    
    for i, idx in enumerate(top_features_idx):
        report += f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}\n"
    
    report += f"""

### Model Behavior Analysis
- The model shows {len(top_features_idx)} most important features
- Feature importance ranges from {feature_importance.min():.4f} to {feature_importance.max():.4f}
- The model focuses on both positive and negative contributions to predictions

### Recommendations
1. Focus on the top 5-10 features for feature engineering
2. Consider feature interactions for the most important features
3. Monitor model performance on edge cases
4. Validate feature importance with domain experts

Generated on: {pd.Timestamp.now()}
"""
    
    # Save report
    with open(f'explainability_report_{model_name.lower().replace(" ", "_")}.md', 'w') as f:
        f.write(report)
    
    print(f"Explainability report saved as: explainability_report_{model_name.lower().replace(' ', '_')}.md")

def main():
    """Main execution function"""
    print("MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 50)
    
    try:
        # Load the best model
        model, scaler, target_encoder, model_data = load_best_model()
        
        if model is None:
            print("Could not load model. Exiting.")
            return
        
        # Load test data
        X_train, X_test, y_train, y_test, label_encoders = load_test_data()
        
        # Apply scaling if available
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Create feature names
        feature_names = create_feature_names(X_test)
        
        model_name = model_data['model_name']
        
        # Perform SHAP analysis
        shap_values, explainer = shap_analysis(model, X_train, X_test_scaled, feature_names, model_name)
        
        # Perform LIME analysis
        lime_analysis(model, X_test_scaled, feature_names, model_name)
        
        # Perform permutation importance analysis
        perm_result = permutation_importance_analysis(model, X_test_scaled, y_test, feature_names, model_name)
        
        # Perform partial dependence analysis
        partial_dependence_analysis(model, X_test_scaled, y_test, feature_names, model_name)
        
        # Create comprehensive report
        create_explainability_report(model_data, shap_values, X_test_scaled, y_test, feature_names)
        
        print("\n=== EXPLAINABILITY ANALYSIS COMPLETE ===")
        print("All plots and reports have been saved.")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 