"""
Data Utilities for Machine Learning Models

This module provides helper functions for data loading, preprocessing,
and splitting for machine learning tasks.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def load_data(filepath, target_column=None, **kwargs):
    """
    Load data from a file (CSV, Excel, etc.) into a pandas DataFrame.
    
    Args:
        filepath: Path to the data file
        target_column: Name of the target/label column
        **kwargs: Additional arguments to pass to pd.read_csv or pd.read_excel
        
    Returns:
        X, y (if target_column is provided) or DataFrame (if target_column is None)
    """
    # Determine file type from extension
    file_extension = filepath.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(filepath, **kwargs)
    elif file_extension in ['xls', 'xlsx', 'xlsm']:
        df = pd.read_excel(filepath, **kwargs)
    elif file_extension == 'json':
        df = pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the data")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    
    return df


def split_data(X, y, test_size=0.2, val_size=0.0, random_state=None, stratify=None):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X: Features
        y: Target/labels
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        stratify: If not None, data is split in a stratified fashion using this as the class labels
        
    Returns:
        If val_size > 0: X_train, X_val, X_test, y_train, y_val, y_test
        If val_size = 0: X_train, X_test, y_train, y_test
    """
    if val_size > 0:
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        # Second split: separate validation set from remaining data
        # Adjust validation size to account for the already removed test set
        adjusted_val_size = val_size / (1 - test_size)
        
        # For stratification in the second split
        if stratify is not None:
            stratify_temp = y_temp
        else:
            stratify_temp = None
            
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=stratify_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        # Simple train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        return X_train, X_test, y_train, y_test


def preprocess_features(X_train, X_test=None, categorical_cols=None, numerical_cols=None, 
                       scaling='standard', handle_missing=True):
    """
    Preprocess features by handling missing values, encoding categorical variables,
    and scaling numerical variables.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        scaling: Type of scaling to apply to numerical features ('standard', 'minmax', or None)
        handle_missing: Whether to impute missing values
        
    Returns:
        Preprocessed X_train and X_test (if provided)
    """
    # Convert to DataFrame if numpy array
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    
    if X_test is not None and isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    
    # Auto-detect column types if not provided
    if categorical_cols is None and numerical_cols is None:
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create copies to avoid modifying the original data
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy() if X_test is not None else None
    
    # Handle missing values
    if handle_missing:
        # For numerical columns
        if numerical_cols:
            num_imputer = SimpleImputer(strategy='mean')
            X_train_processed[numerical_cols] = num_imputer.fit_transform(X_train_processed[numerical_cols])
            
            if X_test_processed is not None:
                X_test_processed[numerical_cols] = num_imputer.transform(X_test_processed[numerical_cols])
        
        # For categorical columns
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train_processed[categorical_cols] = cat_imputer.fit_transform(X_train_processed[categorical_cols])
            
            if X_test_processed is not None:
                X_test_processed[categorical_cols] = cat_imputer.transform(X_test_processed[categorical_cols])
    
    # Encode categorical variables
    encoders = {}
    if categorical_cols:
        for col in categorical_cols:
            encoder = OneHotEncoder(sparse=False, drop='first')
            # Reshape for 2D input
            train_encoded = encoder.fit_transform(X_train_processed[[col]])
            
            # Create feature names
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
            
            # Convert to DataFrame with proper column names
            train_encoded_df = pd.DataFrame(train_encoded, index=X_train_processed.index, columns=feature_names)
            
            # Store the encoder for transform
            encoders[col] = (encoder, feature_names)
            
            # Drop the original column and add encoded columns
            X_train_processed = pd.concat([X_train_processed.drop(columns=[col]), train_encoded_df], axis=1)
            
            if X_test_processed is not None:
                test_encoded = encoder.transform(X_test_processed[[col]])
                test_encoded_df = pd.DataFrame(test_encoded, index=X_test_processed.index, columns=feature_names)
                X_test_processed = pd.concat([X_test_processed.drop(columns=[col]), test_encoded_df], axis=1)
    
    # Scale numerical variables
    if scaling and numerical_cols:
        if scaling == 'standard':
            scaler = StandardScaler()
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling}")
        
        X_train_processed[numerical_cols] = scaler.fit_transform(X_train_processed[numerical_cols])
        
        if X_test_processed is not None:
            X_test_processed[numerical_cols] = scaler.transform(X_test_processed[numerical_cols])
    
    if X_test_processed is not None:
        return X_train_processed, X_test_processed
    else:
        return X_train_processed


def preprocess_target(y_train, y_test=None, task_type='classification'):
    """
    Preprocess target variable by encoding categorical targets.
    
    Args:
        y_train: Training targets
        y_test: Test targets (optional)
        task_type: Task type ('classification' or 'regression')
        
    Returns:
        Preprocessed y_train and y_test (if provided), and encoder (if classification)
    """
    if task_type == 'regression':
        # No preprocessing needed for regression targets
        if y_test is not None:
            return y_train, y_test, None
        else:
            return y_train, None
    
    elif task_type == 'classification':
        # Encode categorical targets
        encoder = LabelEncoder()
        y_train_processed = encoder.fit_transform(y_train)
        
        if y_test is not None:
            y_test_processed = encoder.transform(y_test)
            return y_train_processed, y_test_processed, encoder
        else:
            return y_train_processed, encoder
    
    else:
        raise ValueError(f"Unknown task type: {task_type}") 