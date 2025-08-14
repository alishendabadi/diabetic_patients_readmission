#!/usr/bin/env python3
"""
Test script for balancing functionality
"""

import numpy as np
import pandas as pd
from collections import Counter

def test_balancing_functions():
    """Test the balancing functions with synthetic data"""
    print("Testing balancing functions...")
    
    # Create synthetic imbalanced dataset
    np.random.seed(42)
    
    # Generate synthetic data with 3 classes
    n_samples = 1000
    n_features = 5
    
    # Create imbalanced classes: 70% class 0, 20% class 1, 10% class 2
    y = np.concatenate([
        np.zeros(700),  # Class 0
        np.ones(200),   # Class 1  
        np.full(100, 2) # Class 2
    ])
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    print(f"Original dataset shape: {X.shape}")
    print("Original class distribution:")
    for class_label, count in Counter(y).items():
        print(f"  Class {class_label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Test undersampling
    print("\n--- Testing Undersampling ---")
    try:
        from diabetic_readmission_classification import apply_undersampling
        X_under, y_under = apply_undersampling(X, y)
        print("Undersampling successful!")
        print("Balanced class distribution:")
        for class_label, count in Counter(y_under).items():
            print(f"  Class {class_label}: {count} ({count/len(y_under)*100:.1f}%)")
    except Exception as e:
        print(f"Undersampling failed: {e}")
    
    # Test SMOTE
    print("\n--- Testing SMOTE ---")
    try:
        from diabetic_readmission_classification import apply_smote
        X_smote, y_smote = apply_smote(X, y)
        print("SMOTE successful!")
        print("Balanced class distribution:")
        for class_label, count in Counter(y_smote).items():
            print(f"  Class {class_label}: {count} ({count/len(y_smote)*100:.1f}%)")
    except Exception as e:
        print(f"SMOTE failed: {e}")
    
    print("\nBalancing functions test completed!")

if __name__ == "__main__":
    test_balancing_functions() 