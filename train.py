import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
import yaml

def load_params():
    """Load parameters from params.yaml
    
    This function reads the params.yaml file and extracts the 'train' section.
    DVC tracks these parameters in dvc.yaml under the 'params' field.
    When these values change, DVC will automatically re-run the train stage.
    """
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        return params['train']
    except FileNotFoundError:
        print("params.yaml not found. Using default parameters.")
        return {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42
        }

def load_train_data(filepath):
    """Load training data"""
    print(f"Loading training data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Training data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df

def prepare_features(df):
    """Prepare features and target"""
    # Assume last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X, y, params):
    """Train a Logistic Regression classifier"""
    print("Training Logistic Regression model...")
    print(f"Parameters: {params}")
    
    model = LogisticRegression(
        C=params.get('C', 1.0),
        max_iter=params.get('max_iter', 1000),
        solver=params.get('solver', 'lbfgs'),
        random_state=params.get('random_state', 42)
    )
    
    model.fit(X, y)
    print("Model training completed!")
    
    return model

def save_model(model, scaler, model_path='models/model.pkl', scaler_path='models/scaler.pkl'):
    """Save the trained model and scaler"""
    os.makedirs("models", exist_ok=True)
    
    print(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Saving scaler to {scaler_path}...")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully!")

def main():
    # Configuration
    train_data_path = "data/processed/train.csv"
    
    # Load parameters
    params = load_params()
    
    # Load training data
    train_df = load_train_data(train_data_path)
    
    # Prepare features
    X_train, y_train, scaler = prepare_features(train_df)
    
    # Train model
    model = train_model(X_train, y_train, params)
    
    # Save model and scaler
    save_model(model, scaler)
    
    # Print feature importance
    if hasattr(model, 'coef_'):
        print("\nTop 10 Feature Coefficients:")
        feature_names = train_df.columns[:-1]
        coefficients = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        }).sort_values('coefficient', ascending=False, key=abs)
        print(coefficients.head(10).to_string(index=False))
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()