import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import os

def load_model_and_scaler(model_path='models/model.pkl', scaler_path='models/scaler.pkl'):
    """Load the trained model and scaler"""
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def load_test_data(filepath):
    """Load test data"""
    print(f"Loading test data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Test data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df

def prepare_test_features(df, scaler):
    """Prepare test features"""
    # Assume last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Standardize features using the fitted scaler
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and compute metrics"""
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    }
    
    # Add ROC AUC for binary classification
    if len(np.unique(y_test)) == 2:
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
        except:
            pass
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, output_path='metrics/confusion_matrix.png'):
    """Create and save confusion matrix plot"""
    print(f"\nGenerating confusion matrix plot...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

def save_metrics(metrics, output_path='metrics/metrics.json'):
    """Save metrics to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nSaving metrics to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Metrics saved successfully!")

def print_classification_report(y_test, y_pred):
    """Print detailed classification report"""
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

def main():
    # Configuration
    test_data_path = "data/processed/test.csv"
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Load test data
    test_df = load_test_data(test_data_path)
    
    # Prepare test features
    X_test, y_test = prepare_test_features(test_df, scaler)
    
    # Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Print detailed classification report
    print_classification_report(y_test, y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Save metrics
    save_metrics(metrics)
    
    print("\nValidation pipeline completed successfully!")

if __name__ == "__main__":
    main()