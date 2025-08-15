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
from diabetic_readmission_classification import handle_missing_values
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
    else:
        print("Available Models:")
        for model_file in model_files:
            print(model_file)
    
    # # Load all models and find the best one based on recall_30
    # best_model_data = None
    # best_recall_30 = -1
    
    model_file = input("Enter the name of model file you want to run XAI script on: ")
    with open(os.path.join('models', model_file), 'rb') as f:
        model_data = pickle.load(f)
        
        recall_30 = model_data['metrics']['recall_30']

        best_recall_30 = recall_30
        best_model_data = model_data
    
    if best_model_data is None:
        print("No valid models found.")
        return None, None, None, None
    
    print(f"Model: {best_model_data['model_name']}")
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
    print("Handling missing values...")
    
    df_processed, categorical_columns, numerical_columns = handle_missing_values(df_processed)
    
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

# def create_feature_names(X):
#     """Create meaningful feature names"""
#     feature_names = []
#     for i in range(X.shape[1]):
#         feature_names.append(f'Feature_{i+1}')
#     return feature_names

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
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    return shap_values, explainer

def shap_plot (shap_values, explainer, X_test, feature_names, model_name):
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                     plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'shap_importance_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

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
        feature_names = model_data['feature_names']
        
        model_name = model_data['model_name']
        
        # Perform SHAP analysis
        shap_values, explainer = shap_analysis(model, X_train, X_test, feature_names, model_name)
        shap_plot(shap_values, explainer, X_test, feature_names, model_name)
        print("\n=== EXPLAINABILITY ANALYSIS COMPLETE ===")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
