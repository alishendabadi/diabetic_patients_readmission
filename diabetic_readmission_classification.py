#!/usr/bin/env python3
"""
Diabetic Patient Readmission Classification
==========================================

This script performs classification of diabetic patients based on their readmission status:
- NO: No readmission
- <30: Readmitted within 30 days  
- >30: Readmitted after 30 days

The focus is on maximizing Recall for the <30 days class.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score,
                           roc_curve, precision_recall_curve)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from tqdm.auto import tqdm
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_explore_data():
    """Load and explore the diabetic dataset"""
    print("Loading data...")
    df = pd.read_csv('diabetic2.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Display basic info
    print("\n=== DATASET OVERVIEW ===")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    # Check target variable
    print("\n=== TARGET VARIABLE DISTRIBUTION ===")
    print(df['readmitted'].value_counts())
    print(f"\nTarget variable distribution (%):")
    print(df['readmitted'].value_counts(normalize=True) * 100)
    
    # Check missing values
    print("\n=== MISSING VALUES ===")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percent': missing_percent
    }).sort_values('Missing Count', ascending=False)
    print(missing_df[missing_df['Missing Count'] > 0])
    
    return df

def preprocess_data(df):
    """Preprocess the data including handling missing values and encoding"""
    print("\n=== PREPROCESSING DATA ===")
    
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Handle missing values
    print("Handling missing values...")
    
    # For categorical columns, fill with 'Unknown'
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna('Unknown', inplace=True)
    
    # For numerical columns, fill with median
    numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    label_encoders = {}
    
    for col in categorical_columns:
        if col != 'readmitted':  # Don't encode target yet
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    df_processed['readmitted_encoded'] = target_encoder.fit_transform(df_processed['readmitted'])
    
    print("Target encoding:")
    for i, label in enumerate(target_encoder.classes_):
        print(f"  {i}: {label}")
    
    # Remove unnecessary columns
    columns_to_drop = ['encounter_id', 'patient_nbr', 'readmitted']
    df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"Final dataset shape after preprocessing: {df_processed.shape}")
    
    return df_processed, target_encoder

def apply_undersampling(X_train, y_train):
    """Apply undersampling to balance the dataset"""
    print("Applying Random Undersampling...")
    
    # Combine features and target
    df_train = pd.DataFrame(X_train)
    df_train['target'] = y_train
    
    # Find the minority class
    class_counts = df_train['target'].value_counts()
    minority_class = class_counts.idxmin()
    minority_count = class_counts.min()
    
    print(f"Minority class: {minority_class} with {minority_count} samples")
    
    # Undersample each class to match the minority class
    balanced_dfs = []
    
    for class_label in class_counts.index:
        class_df = df_train[df_train['target'] == class_label]
        
        if len(class_df) > minority_count:
            # Undersample to minority count
            undersampled_df = resample(class_df,
                                     replace=False,
                                     n_samples=minority_count,
                                     random_state=42)
        else:
            # Keep as is if already minority
            undersampled_df = class_df
        
        balanced_dfs.append(undersampled_df)
    
    # Combine all balanced classes
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the data
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate features and target
    X_balanced = balanced_df.drop('target', axis=1).values
    y_balanced = balanced_df['target'].values
    
    print(f"Balanced dataset shape: {X_balanced.shape}")
    
    return X_balanced, y_balanced

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance the dataset"""
    print("Applying SMOTE...")
    
    # Initialize SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    
    # Apply SMOTE
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original dataset shape: {X_train.shape}")
    print(f"Balanced dataset shape: {X_balanced.shape}")
    
    return X_balanced, y_balanced

def prepare_features(df_processed, balance_method='none'):
    """Prepare features and target for modeling with optional balancing"""
    print("\n=== PREPARING FEATURES ===")
    
    # Separate features and target
    X = df_processed.drop('readmitted_encoded', axis=1)
    y = df_processed['readmitted_encoded']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Display class distribution before balancing
    print("\nClass distribution before balancing:")
    unique, counts = np.unique(y, return_counts=True)
    for i, (class_label, count) in enumerate(zip(unique, counts)):
        percentage = (count / len(y)) * 100
        print(f"  Class {class_label}: {count} ({percentage:.1f}%)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    if balance_method != 'none':
        print(f"\n=== APPLYING {balance_method.upper()} BALANCING ===")
        
        if balance_method == 'undersample':
            X_train_balanced, y_train_balanced = apply_undersampling(X_train_scaled, y_train)
        elif balance_method == 'smote':
            X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)
        else:
            print(f"Unknown balancing method: {balance_method}. Using original data.")
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        # Display class distribution after balancing
        print("\nClass distribution after balancing:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        for i, (class_label, count) in enumerate(zip(unique, counts)):
            percentage = (count / len(y_train_balanced)) * 100
            print(f"  Class {class_label}: {count} ({percentage:.1f}%)")
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_model(model, model_name, balance_method, scaler, target_encoder, metrics):
    """Save a trained model and related components as pickle files"""
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create a unique filename
    filename = f"models/{model_name}_{balance_method}_model.pkl"
    
    # Save model and related components
    model_data = {
        'model': model,
        'scaler': scaler,
        'target_encoder': target_encoder,
        'model_name': model_name,
        'balance_method': balance_method,
        'metrics': metrics,
        'timestamp': pd.Timestamp.now()
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved as: {filename}")
    return filename

def train_models(X_train, X_test, y_train, y_test, balance_method='none', scaler=None, target_encoder=None):
    """Train multiple classification models"""
    print("\n=== TRAINING MODELS ===")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    results = {}
    
    for name, model in tqdm(models.items(), desc=f"Training models [{balance_method}]", leave=True):
        tqdm.write(f"Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate recall for <30 days class (assuming it's class 1)
        recall_30 = recall_score(y_test, y_pred, labels=[1], average=None)[0] if 1 in y_test.unique() else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'recall_30': recall_30
        }
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'recall_30': recall_30,
            'balance_method': balance_method
        }
        
        tqdm.write(f"  Accuracy: {accuracy:.4f}")
        tqdm.write(f"  Precision: {precision:.4f}")
        tqdm.write(f"  Recall: {recall:.4f}")
        tqdm.write(f"  F1-Score: {f1:.4f}")
        tqdm.write(f"  Recall for <30 days: {recall_30:.4f}")
        tqdm.write("")
        
        # Save each model
        if scaler is not None and target_encoder is not None:
            save_model(model, name, balance_method, scaler, target_encoder, metrics)
    
    return results

def evaluate_models(results, y_test, target_encoder):
    """Evaluate and compare all models"""
    print("\n=== MODEL EVALUATION ===")
    
    # Create comparison table
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1'],
            'Recall_<30': result['recall_30']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("Model Performance Comparison:")
    print(comparison_df.round(4))
    
    # Find best model for <30 days recall
    best_model_name = comparison_df.loc[comparison_df['Recall_<30'].idxmax(), 'Model']
    print(f"\nBest model for <30 days recall: {best_model_name}")
    
    return comparison_df, best_model_name

def plot_results(results, y_test, target_encoder):
    """Create comprehensive visualization plots"""
    print("\n=== CREATING PLOTS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        values = [results[name][metric.lower().replace('-', '_')] for name in results.keys()]
        bars = ax.bar(results.keys(), values, alpha=0.7)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrices', fontsize=16)
    
    class_names = target_encoder.classes_
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i//3, i%3]
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide the last subplot if needed
    if len(results) < 6:
        axes[-1, -1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Feature Importance (for tree-based models)
    tree_models = ['Random Forest', 'XGBoost']
    if any(name in results for name in tree_models):
        fig, axes = plt.subplots(1, len(tree_models), figsize=(15, 6))
        fig.suptitle('Feature Importance', fontsize=16)
        
        if len(tree_models) == 1:
            axes = [axes]
        
        for i, name in enumerate(tree_models):
            if name in results:
                model = results[name]['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = [f'Feature_{j}' for j in range(len(importances))]
                    
                    # Get top 15 features
                    indices = np.argsort(importances)[::-1][:15]
                    
                    axes[i].bar(range(len(indices)), importances[indices])
                    axes[i].set_title(f'{name} - Top 15 Features')
                    axes[i].set_xlabel('Features')
                    axes[i].set_ylabel('Importance')
                    axes[i].set_xticks(range(len(indices)))
                    axes[i].set_xticklabels([feature_names[j] for j in indices], rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, result in results.items():
        # Calculate ROC for each class
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), result['y_pred_proba'][:, i])
            auc = roc_auc_score((y_test == i).astype(int), result['y_pred_proba'][:, i])
            ax.plot(fpr, tpr, label=f'{name} - {class_name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, result in results.items():
        # Calculate PR for <30 days class (assuming it's class 1)
        if 1 in y_test.unique():
            precision, recall, _ = precision_recall_curve((y_test == 1).astype(int), 
                                                        result['y_pred_proba'][:, 1])
            ax.plot(recall, precision, label=f'{name}')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves (Focus on <30 days class)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_balancing_comparison(all_results, target_encoder):
    """Create comparison plots for different balancing methods"""
    print("\n=== CREATING BALANCING COMPARISON PLOTS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Recall for <30 days comparison across all methods
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = list(all_results.keys())
    models = list(all_results[methods[0]]['results'].keys())
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, method in enumerate(methods):
        recalls = [all_results[method]['results'][model]['recall_30'] for model in models]
        ax.bar(x + i*width, recalls, width, label=method.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Recall for <30 Days')
    ax.set_title('Recall for <30 Days Class Across Different Balancing Methods')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('balancing_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Overall accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        accuracies = [all_results[method]['results'][model]['accuracy'] for model in models]
        ax.bar(x + i*width, accuracies, width, label=method.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Overall Accuracy Across Different Balancing Methods')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('balancing_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. F1-score comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        f1_scores = [all_results[method]['results'][model]['f1'] for model in models]
        ax.bar(x + i*width, f1_scores, width, label=method.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score Across Different Balancing Methods')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('balancing_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Heatmap of all metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['recall_30', 'accuracy', 'f1']
    metric_names = ['Recall (<30)', 'Accuracy', 'F1-Score']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        # Create matrix for heatmap
        matrix = []
        for method in methods:
            row = [all_results[method]['results'][model][metric] for model in models]
            matrix.append(row)
        
        # Create heatmap
        im = axes[i].imshow(matrix, cmap='YlOrRd', aspect='auto')
        axes[i].set_xticks(range(len(models)))
        axes[i].set_yticks(range(len(methods)))
        axes[i].set_xticklabels(models, rotation=45)
        axes[i].set_yticklabels([m.capitalize() for m in methods])
        axes[i].set_title(f'{metric_name} Heatmap')
        
        # Add text annotations
        for j in range(len(methods)):
            for k in range(len(models)):
                text = axes[i].text(k, j, f'{matrix[j][k]:.3f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('balancing_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def detailed_classification_report(results, y_test, target_encoder):
    """Print detailed classification reports"""
    print("\n=== DETAILED CLASSIFICATION REPORTS ===")
    
    for name, result in results.items():
        print(f"\n{name} Classification Report:")
        print("=" * 50)
        print(classification_report(y_test, result['y_pred'], 
                                  target_names=target_encoder.classes_))
        
        # Special focus on <30 days class
        if 1 in y_test.unique():
            print(f"\n{name} - Focus on <30 days class:")
            print(f"Recall for <30 days: {result['recall_30']:.4f}")
            print()

def main():
    """Main execution function"""
    print("DIABETIC PATIENT READMISSION CLASSIFICATION")
    print("=" * 50)
    
    try:
        # Load and explore data
        df = load_and_explore_data()
        
        # Preprocess data
        df_processed, target_encoder = preprocess_data(df)
        
        # Define balancing methods to test
        balance_methods = ['none', 'undersample', 'smote']
        all_results = {}
        all_comparisons = []
        
        for balance_method in balance_methods:
            print(f"\n{'='*60}")
            print(f"TESTING WITH {balance_method.upper()} BALANCING")
            print(f"{'='*60}")
            
            # Prepare features with current balancing method
            X_train, X_test, y_train, y_test, scaler = prepare_features(df_processed, balance_method)
            
            # Train models
            results = train_models(X_train, X_test, y_train, y_test, balance_method, scaler, target_encoder)
            
            # Evaluate models
            comparison_df, best_model = evaluate_models(results, y_test, target_encoder)
            
            # Add balancing method to comparison
            comparison_df['Balance_Method'] = balance_method
            all_comparisons.append(comparison_df)
            
            # Store results
            all_results[balance_method] = {
                'results': results,
                'comparison_df': comparison_df,
                'best_model': best_model,
                'X_test': X_test,
                'y_test': y_test
            }
            
            # Print detailed reports for this method
            detailed_classification_report(results, y_test, target_encoder)
        
        # Compare all methods
        print(f"\n{'='*60}")
        print("COMPARISON OF ALL BALANCING METHODS")
        print(f"{'='*60}")
        
        # Combine all comparisons
        combined_comparison = pd.concat(all_comparisons, ignore_index=True)
        
        # Display comparison table
        print("\nOverall Performance Comparison:")
        print(combined_comparison.round(4))
        
        # Find best overall model for <30 days recall
        best_overall_idx = combined_comparison['Recall_<30'].idxmax()
        best_overall = combined_comparison.loc[best_overall_idx]
        
        print(f"\nBest overall model for <30 days recall:")
        print(f"Model: {best_overall['Model']}")
        print(f"Balance Method: {best_overall['Balance_Method']}")
        print(f"Recall for <30 days: {best_overall['Recall_<30']:.4f}")
        print(f"Overall Accuracy: {best_overall['Accuracy']:.4f}")
        
        # Create comparison plots
        plot_balancing_comparison(all_results, target_encoder)
        
        # Save results
        combined_comparison.to_csv('model_comparison_results.csv', index=False)
        print("\nResults saved to 'model_comparison_results.csv'")
        print("Plots saved as PNG files")
        
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Best overall model: {best_overall['Model']} with {best_overall['Balance_Method']} balancing")
        print(f"Best recall for <30 days: {best_overall['Recall_<30']:.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 