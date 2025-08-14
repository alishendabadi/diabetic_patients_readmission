#!/usr/bin/env python3
"""
Load and Use Saved Model
=======================

This script demonstrates how to load a saved model and use it for predictions.
"""

import pickle
import pandas as pd
import numpy as np
import os

def load_model(model_path):
    """Load a saved model from pickle file"""
    print(f"Loading model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Model loaded successfully!")
    print(f"Model: {model_data['model_name']}")
    print(f"Balance method: {model_data['balance_method']}")
    print(f"Performance metrics:")
    for metric, value in model_data['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    return model_data

def prepare_sample_data():
    """Prepare sample data for prediction (same preprocessing as training)"""
    print("Preparing sample data...")
    
    # Load original data to get a sample
    df = pd.read_csv('diabetic2.csv')
    
    # Take a few samples for demonstration
    sample_df = df.head(5).copy()
    
    # Apply same preprocessing as in main script
    # Handle missing values
    categorical_columns = sample_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if sample_df[col].isnull().sum() > 0:
            sample_df[col].fillna('Unknown', inplace=True)
    
    numerical_columns = sample_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        if sample_df[col].isnull().sum() > 0:
            sample_df[col].fillna(sample_df[col].median(), inplace=True)
    
    # Encode categorical variables (using same encoders as training)
    from sklearn.preprocessing import LabelEncoder
    
    for col in categorical_columns:
        if col != 'readmitted':
            le = LabelEncoder()
            sample_df[col] = le.fit_transform(sample_df[col].astype(str))
    
    # Encode target variable
    target_encoder = LabelEncoder()
    sample_df['readmitted_encoded'] = target_encoder.fit_transform(sample_df['readmitted'])
    
    # Remove unnecessary columns
    columns_to_drop = ['encounter_id', 'patient_nbr', 'readmitted']
    sample_df = sample_df.drop(columns=columns_to_drop, errors='ignore')
    
    # Separate features and target
    X_sample = sample_df.drop('readmitted_encoded', axis=1)
    y_sample = sample_df['readmitted_encoded']
    
    print(f"Sample data shape: {X_sample.shape}")
    print(f"Sample target values: {y_sample.values}")
    
    return X_sample, y_sample, target_encoder

def make_predictions(model_data, X_sample):
    """Make predictions using the loaded model"""
    print("\nMaking predictions...")
    
    model = model_data['model']
    scaler = model_data['scaler']
    target_encoder = model_data['target_encoder']
    
    # Apply scaling if available
    if scaler is not None:
        X_scaled = scaler.transform(X_sample)
        print("Applied feature scaling")
    else:
        X_scaled = X_sample
        print("No scaling applied")
    
    # Make predictions
    predictions = model.predict(X_scaled)
    prediction_proba = model.predict_proba(X_scaled)
    
    # Decode predictions
    predicted_labels = target_encoder.inverse_transform(predictions)
    
    print("\nPrediction Results:")
    print("=" * 50)
    
    for i in range(len(predictions)):
        print(f"\nSample {i+1}:")
        print(f"  Predicted class: {predictions[i]} ({predicted_labels[i]})")
        print(f"  Prediction probabilities:")
        for j, prob in enumerate(prediction_proba[i]):
            class_name = target_encoder.inverse_transform([j])[0]
            print(f"    {class_name}: {prob:.4f}")
    
    return predictions, prediction_proba, predicted_labels

def main():
    """Main execution function"""
    print("LOAD AND USE SAVED MODEL")
    print("=" * 50)
    
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            print("Models directory not found. Please run the main training script first.")
            return
        
        # List available models
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        
        if not model_files:
            print("No model files found in models directory.")
            return
        
        print(f"Available models:")
        for i, model_file in enumerate(model_files):
            print(f"  {i+1}. {model_file}")
        
        # Load the first model (you can modify this to load a specific model)
        model_path = os.path.join('models', model_files[0])
        model_data = load_model(model_path)
        
        # Prepare sample data
        X_sample, y_sample, target_encoder = prepare_sample_data()
        
        # Make predictions
        predictions, prediction_proba, predicted_labels = make_predictions(model_data, X_sample)
        
        print(f"\n=== PREDICTION COMPLETE ===")
        print(f"Model used: {model_data['model_name']}")
        print(f"Number of samples predicted: {len(predictions)}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 