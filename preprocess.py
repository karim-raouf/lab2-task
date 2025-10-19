import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(filepath):
    """Load the raw CSV dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    """Clean the dataset"""
    print("Cleaning data...")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Handle missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values found:")
        print(missing_counts[missing_counts > 0])
        
        # Drop rows with missing target variable
        target_col = df.columns[-1]
        df = df.dropna(subset=[target_col])
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def encode_features(df):
    """Encode categorical variables"""
    print("Encoding categorical features...")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders

def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print(f"Splitting data (test_size={test_size})...")
    
    # Assume last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Combine back into DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df

def main():
    # Configuration
    raw_data_path = "data/raw/heart.csv"
    train_output_path = "data/processed/train.csv"
    test_output_path = "data/processed/test.csv"
    
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Load data
    df = load_data(raw_data_path)
    
    # Clean data
    df = clean_data(df)
    
    # Encode features
    df, label_encoders = encode_features(df)
    
    # Split data
    train_df, test_df = split_data(df)
    
    # Save processed data
    print(f"Saving train data to {train_output_path}...")
    train_df.to_csv(train_output_path, index=False)
    
    print(f"Saving test data to {test_output_path}...")
    test_df.to_csv(test_output_path, index=False)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()