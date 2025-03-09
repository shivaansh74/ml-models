"""
Gradient Boosting Example: Customer Churn Prediction

This example demonstrates how to use the Gradient Boosting model for 
customer churn prediction. The example shows:

1. Loading and exploring a synthetic customer churn dataset
2. Preprocessing the data and handling categorical features
3. Building and training a Gradient Boosting model
4. Evaluating model performance and tuning hyperparameters
5. Analyzing feature importance to understand key churn factors
6. Making predictions on new customer data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.datasets import make_classification
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.ensemble.gradient_boosting import GradientBoostingModel


def create_synthetic_churn_data(n_samples=1000):
    """
    Create a synthetic customer churn dataset.
    
    Args:
        n_samples: Number of customer records to generate
        
    Returns:
        DataFrame with customer features and churn label
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f'CUST_{i:06d}' for i in range(n_samples)]
    
    # Generate age data (between 18 and 85)
    age = np.random.randint(18, 86, size=n_samples)
    
    # Generate gender (0: Male, 1: Female)
    gender = np.random.randint(0, 2, size=n_samples)
    
    # Generate subscription length in months (1 to 60)
    tenure = np.random.randint(1, 61, size=n_samples)
    
    # Generate monthly charges ($20 to $120)
    monthly_charges = np.random.uniform(20, 120, size=n_samples)
    
    # Generate total charges (monthly_charges * tenure with some variation)
    total_charges = monthly_charges * tenure * (1 + np.random.uniform(-0.1, 0.1, size=n_samples))
    
    # Contract type (0: Month-to-month, 1: One year, 2: Two year)
    contract = np.random.randint(0, 3, size=n_samples)
    
    # Online security (0: No, 1: Yes)
    online_security = np.random.randint(0, 2, size=n_samples)
    
    # Tech support (0: No, 1: Yes)
    tech_support = np.random.randint(0, 2, size=n_samples)
    
    # Internet service (0: DSL, 1: Fiber optic, 2: None)
    internet_service = np.random.randint(0, 3, size=n_samples)
    
    # Payment method (0: Electronic check, 1: Mailed check, 2: Bank transfer, 3: Credit card)
    payment_method = np.random.randint(0, 4, size=n_samples)
    
    # Customer satisfaction score (1-5)
    satisfaction_score = np.random.randint(1, 6, size=n_samples)
    
    # Create a feature matrix for generating the churn target
    X_for_churn = np.column_stack([
        age / 85,  # Normalize to 0-1
        gender,
        tenure / 60,  # Normalize to 0-1
        monthly_charges / 120,  # Normalize to 0-1
        contract / 2,  # Normalize to 0-1
        online_security,
        tech_support,
        internet_service / 2,  # Normalize to 0-1
        satisfaction_score / 5  # Normalize to 0-1
    ])
    
    # Generate churn based on the above features with realistic relationships
    # Lower tenure, month-to-month contracts, higher monthly charges,
    # no online security or tech support, fiber optic internet, and lower satisfaction
    # are all associated with higher churn rates
    
    # Coefficients for each feature's contribution to churn
    # Negative values mean the feature reduces churn
    feature_weights = np.array([
        -0.2,   # Age (older customers slightly less likely to churn)
        0.0,    # Gender (no significant effect)
        -1.5,   # Tenure (longer tenure much less likely to churn)
        0.7,    # Monthly charges (higher charges more likely to churn)
        -1.2,   # Contract (longer contracts less likely to churn)
        -0.8,   # Online security (having security less likely to churn)
        -0.8,   # Tech support (having support less likely to churn)
        0.6,    # Internet service (fiber more likely to churn than DSL)
        -1.5    # Satisfaction (higher satisfaction much less likely to churn)
    ])
    
    # Calculate churn probabilities
    churn_scores = X_for_churn.dot(feature_weights)
    churn_probs = 1 / (1 + np.exp(-churn_scores))
    
    # Generate actual churn based on probabilities
    churn = (np.random.random(size=n_samples) < churn_probs).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': age,
        'Gender': ['Male' if g == 0 else 'Female' for g in gender],
        'Tenure': tenure,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'Contract': ['Month-to-month' if c == 0 else 'One year' if c == 1 else 'Two year' for c in contract],
        'OnlineSecurity': ['No' if o == 0 else 'Yes' for o in online_security],
        'TechSupport': ['No' if t == 0 else 'Yes' for t in tech_support],
        'InternetService': ['DSL' if i == 0 else 'Fiber optic' if i == 1 else 'None' for i in internet_service],
        'PaymentMethod': ['Electronic check' if p == 0 else 'Mailed check' if p == 1 
                          else 'Bank transfer' if p == 2 else 'Credit card' for p in payment_method],
        'SatisfactionScore': satisfaction_score,
        'Churn': ['No' if c == 0 else 'Yes' for c in churn]
    })
    
    return df


def explore_data(df):
    """
    Explore the dataset and show key statistics.
    
    Args:
        df: DataFrame with customer data
    """
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nChurn Distribution:")
    churn_counts = df['Churn'].value_counts()
    print(churn_counts)
    print(f"Churn Rate: {churn_counts['Yes'] / len(df):.2%}")
    
    # Plot churn distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.show()
    
    # Plot age distribution by churn
    plt.figure(figsize=(12, 5))
    sns.histplot(data=df, x='Age', hue='Churn', bins=20, element='step')
    plt.title('Age Distribution by Churn')
    plt.show()
    
    # Plot tenure vs. monthly charges
    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(data=df, x='Tenure', y='MonthlyCharges', hue='Churn', alpha=0.7)
    plt.title('Tenure vs. Monthly Charges by Churn')
    plt.show()
    
    # Plot churn rate by contract type
    plt.figure(figsize=(12, 6))
    contract_churn = df.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').mean()
    ).reset_index()
    contract_churn.columns = ['Contract', 'Churn Rate']
    
    sns.barplot(data=contract_churn, x='Contract', y='Churn Rate')
    plt.title('Churn Rate by Contract Type')
    plt.ylabel('Churn Rate')
    plt.show()


def preprocess_data(df):
    """
    Preprocess the data for machine learning.
    
    Args:
        df: DataFrame with customer data
        
    Returns:
        X: Features
        y: Target (churn)
        feature_names: Names of the processed features
    """
    # Convert target to binary
    y = (df['Churn'] == 'Yes').astype(int)
    
    # Select features
    features = [
        'Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TotalCharges',
        'Contract', 'OnlineSecurity', 'TechSupport', 'InternetService',
        'PaymentMethod', 'SatisfactionScore'
    ]
    
    X = df[features].copy()
    
    # Identify categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    categorical_feature_names = []
    for i, col in enumerate(categorical_features):
        categories = preprocessor.transformers_[1][1].categories_[i][1:]  # drop first
        categorical_feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    feature_names = numerical_features + categorical_feature_names
    
    return X_processed, y, feature_names


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Tune hyperparameters for the Gradient Boosting model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        
    Returns:
        best_params: Dictionary of best hyperparameters
    """
    print("\n5. Tuning hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 1.0]
    }
    
    # Use only a subset of parameters to save time
    small_param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 4]
    }
    
    # Create base model
    base_model = GradientBoostingModel(task_type="classification")
    
    # Initialize grid search
    grid_search = GridSearchCV(
        estimator=base_model.model,
        param_grid=small_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"\nBest parameters: {best_params}")
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    val_accuracy = best_model.score(X_val, y_val)
    print(f"Validation accuracy with best parameters: {val_accuracy:.4f}")
    
    return best_params


def main():
    print("=" * 80)
    print("Gradient Boosting Example: Customer Churn Prediction")
    print("=" * 80)
    
    # 1. Load data or create synthetic data
    print("\n1. Loading customer churn dataset...")
    df = create_synthetic_churn_data(n_samples=1000)
    print(f"Created synthetic churn dataset with {len(df)} customers")
    
    # 2. Explore the data
    print("\n2. Exploring the dataset...")
    explore_data(df)
    
    # 3. Preprocess the data
    print("\n3. Preprocessing the data...")
    X, y, feature_names = preprocess_data(df)
    print(f"Processed data shape: {X.shape}")
    print(f"Number of features after preprocessing: {len(feature_names)}")
    
    # 4. Split the data into train, validation, and test sets
    print("\n4. Splitting data into train, validation, and test sets...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # 5. Tune hyperparameters (commented out to save time)
    # best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)
    
    # Use pre-defined "best" parameters to save time
    best_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'min_samples_split': 5,
        'subsample': 0.8
    }
    
    # 6. Train the model with the best parameters
    print("\n6. Training the Gradient Boosting model with selected parameters...")
    gb_model = GradientBoostingModel(
        task_type="classification",
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params.get('min_samples_split', 2),
        subsample=best_params.get('subsample', 1.0),
        random_state=42
    )
    
    # Time the training process
    start_time = time.time()
    gb_model.train(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # 7. Evaluate the model on test data
    print("\n7. Evaluating the model...")
    metrics = gb_model.evaluate(X_test, y_test)
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate predictions for test data
    y_pred = gb_model.predict(X_test)
    y_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Plot confusion matrix
    print("\n8. Visualizing confusion matrix...")
    gb_model.plot_confusion_matrix(X_test, y_test, class_names=['No Churn', 'Churn'])
    
    # 9. Plot ROC curve
    print("\n9. Plotting ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    # 10. Analyze feature importance
    print("\n10. Analyzing feature importance...")
    importances = gb_model.get_feature_importances()
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Plot top 10 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Features by Importance')
    plt.tight_layout()
    plt.show()
    
    # 11. Plot learning curve (score vs. number of trees)
    print("\n11. Plotting learning curve (score vs. number of trees)...")
    gb_model.plot_learning_curve(X_train, y_train, X_val, y_val, 
                              n_estimators_range=np.arange(10, 110, 10))
    
    # 12. Create and predict on a few example customers
    print("\n12. Making predictions on new customer examples...")
    
    # Create a few example customers
    example_customers = pd.DataFrame({
        'CustomerID': ['NEW_001', 'NEW_002', 'NEW_003'],
        'Age': [35, 62, 27],
        'Gender': ['Male', 'Female', 'Male'],
        'Tenure': [2, 48, 6],
        'MonthlyCharges': [89.99, 45.50, 105.75],
        'TotalCharges': [179.98, 2184.0, 634.5],
        'Contract': ['Month-to-month', 'Two year', 'Month-to-month'],
        'OnlineSecurity': ['No', 'Yes', 'No'],
        'TechSupport': ['No', 'Yes', 'No'],
        'InternetService': ['Fiber optic', 'DSL', 'Fiber optic'],
        'PaymentMethod': ['Electronic check', 'Bank transfer', 'Credit card'],
        'SatisfactionScore': [3, 5, 2]
    })
    
    print("\nExample customers:")
    print(example_customers)
    
    # Preprocess the example data (using the same preprocessing as before)
    X_example, _, _ = preprocess_data(example_customers)
    
    # Make predictions
    churn_probs = gb_model.predict_proba(X_example)[:, 1]
    churn_preds = gb_model.predict(X_example)
    
    # Add predictions to the example DataFrame
    example_customers['Churn Probability'] = churn_probs
    example_customers['Predicted Churn'] = ['Yes' if p == 1 else 'No' for p in churn_preds]
    
    print("\nPredictions for example customers:")
    prediction_columns = ['CustomerID', 'Contract', 'Tenure', 'MonthlyCharges', 
                          'OnlineSecurity', 'SatisfactionScore', 
                          'Churn Probability', 'Predicted Churn']
    print(example_customers[prediction_columns])
    
    # 13. Save the model
    print("\n13. Saving the model...")
    model_path = "../models/gradient_boosting_churn_model.joblib"
    gb_model.save_model(model_path)
    
    print("\nExample complete! The Gradient Boosting model has been trained for churn prediction.")


if __name__ == "__main__":
    main() 