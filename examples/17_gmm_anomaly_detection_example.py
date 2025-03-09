"""
Gaussian Mixture Model (GMM) for Anomaly Detection

This example demonstrates the use of Gaussian Mixture Models for anomaly detection
in multivariate data. GMMs model the probability density of normal data points,
making them effective for identifying outliers that don't fit the learned distribution.

The example includes:
1. Synthetic data generation with known anomalies
2. Training a GMM model
3. Anomaly detection using likelihood scores
4. Visualization of clusters and decision boundaries
5. Comparison with other anomaly detection methods
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from scipy import stats

# Add the parent directory to the path so we can import from models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the GMM model
from models.unsupervised.gmm import GMMModel


def create_output_dir(subdir='gmm_anomaly_detection'):
    """Create output directory for models and visualizations."""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', subdir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_synthetic_data(n_samples=1000, n_features=2, contamination=0.05, random_state=42):
    """
    Generate synthetic data with known anomalies.
    
    Args:
        n_samples: Number of data points to generate
        n_features: Number of features
        contamination: Proportion of anomalies
        random_state: Random seed for reproducibility
        
    Returns:
        X: Feature matrix
        y: Labels (0 for normal, 1 for anomaly)
    """
    np.random.seed(random_state)
    
    # Calculate number of normal and anomalous samples
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal
    
    # Generate normal samples from a mixture of Gaussians
    n_gaussian_components = 3
    n_per_component = n_normal // n_gaussian_components
    
    centers = [
        np.random.uniform(-5, 5, n_features),
        np.random.uniform(-5, 5, n_features),
        np.random.uniform(-5, 5, n_features)
    ]
    
    covs = [
        np.eye(n_features) * np.random.uniform(0.5, 1.5),
        np.eye(n_features) * np.random.uniform(0.5, 1.5),
        np.eye(n_features) * np.random.uniform(0.5, 1.5)
    ]
    
    X_normal = np.vstack([
        np.random.multivariate_normal(centers[0], covs[0], n_per_component),
        np.random.multivariate_normal(centers[1], covs[1], n_per_component),
        np.random.multivariate_normal(centers[2], covs[2], n_normal - 2*n_per_component)
    ])
    
    # Generate anomalies
    X_anomalies = []
    for _ in range(n_anomalies):
        # Create anomalies that are outside the normal range
        point = np.random.uniform(-15, 15, n_features)
        # Ensure it's actually an anomaly by checking distance from centers
        while any(np.linalg.norm(point - center) < 8 for center in centers):
            point = np.random.uniform(-15, 15, n_features)
        X_anomalies.append(point)
    
    X_anomalies = np.array(X_anomalies)
    
    # Combine normal and anomalous samples
    X = np.vstack([X_normal, X_anomalies])
    y = np.zeros(n_samples)
    y[n_normal:] = 1  # Label anomalies as 1
    
    # Shuffle the data
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    return X, y


def train_gmm_model(X_train, n_components=3, covariance_type='full'):
    """
    Train a GMM model on the provided data.
    
    Args:
        X_train: Training data
        n_components: Number of mixture components
        covariance_type: Type of covariance parameters
        
    Returns:
        Trained GMM model
    """
    print("\n--- Training GMM Model ---")
    
    # Initialize and train the GMM model
    gmm = GMMModel(
        n_components=n_components,
        covariance_type=covariance_type,
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=5,
        random_state=42
    )
    
    # Train the model
    gmm.train(X_train, scale_data=True)
    
    print(f"GMM model trained with {n_components} components")
    
    return gmm


def evaluate_anomaly_detection(model, X_test, y_test, contamination=0.05):
    """
    Evaluate the performance of the GMM model for anomaly detection.
    
    Args:
        model: Trained GMM model
        X_test: Test data
        y_test: True labels (0 for normal, 1 for anomaly)
        contamination: Expected proportion of anomalies
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n--- Evaluating Anomaly Detection Performance ---")
    
    # Get anomaly scores (negative log-likelihood)
    anomaly_scores = -model.score_samples(X_test)
    
    # Calculate precision-recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, anomaly_scores)
    average_precision = average_precision_score(y_test, anomaly_scores)
    
    # Calculate ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold based on F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_pr[optimal_idx]
    
    # Make predictions using the optimal threshold
    y_pred = (anomaly_scores >= optimal_threshold).astype(int)
    
    # Calculate metrics
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_test)
    precision_optimal = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_optimal = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_optimal = 2 * precision_optimal * recall_optimal / (precision_optimal + recall_optimal) if (precision_optimal + recall_optimal) > 0 else 0
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision_optimal:.4f}")
    print(f"Recall: {recall_optimal:.4f}")
    print(f"F1 Score: {f1_optimal:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    
    # Return evaluation metrics
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision_optimal,
        'recall': recall_optimal,
        'f1_score': f1_optimal,
        'auc_roc': roc_auc,
        'average_precision': average_precision,
        'precision_curve': precision,
        'recall_curve': recall,
        'fpr_curve': fpr,
        'tpr_curve': tpr
    }
    
    return metrics, anomaly_scores


def compare_with_isolation_forest(X_train, X_test, y_test, contamination=0.05):
    """
    Compare GMM anomaly detection with Isolation Forest.
    
    Args:
        X_train: Training data
        X_test: Test data
        y_test: True labels (0 for normal, 1 for anomaly)
        contamination: Expected proportion of anomalies
        
    Returns:
        Isolation Forest anomaly scores and metrics
    """
    print("\n--- Comparing with Isolation Forest ---")
    
    # Train Isolation Forest
    isolation_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    isolation_forest.fit(X_train)
    
    # Get anomaly scores (negated to match GMM scores where higher = more anomalous)
    if_scores = -isolation_forest.score_samples(X_test)
    
    # Calculate precision-recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, if_scores)
    average_precision = average_precision_score(y_test, if_scores)
    
    # Calculate ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, if_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"Isolation Forest - AUC-ROC: {roc_auc:.4f}")
    print(f"Isolation Forest - Average Precision: {average_precision:.4f}")
    
    # Return metrics for comparison
    metrics = {
        'auc_roc': roc_auc,
        'average_precision': average_precision,
        'precision_curve': precision,
        'recall_curve': recall,
        'fpr_curve': fpr,
        'tpr_curve': tpr
    }
    
    return metrics, if_scores


def visualize_anomaly_detection(X_test, y_test, gmm_scores, if_scores, gmm_metrics, if_metrics, threshold):
    """
    Visualize anomaly detection results for 2D data.
    
    Args:
        X_test: Test data
        y_test: True labels
        gmm_scores: Anomaly scores from GMM
        if_scores: Anomaly scores from Isolation Forest
        gmm_metrics: Dictionary of GMM evaluation metrics
        if_metrics: Dictionary of Isolation Forest evaluation metrics
        threshold: Optimal threshold for GMM anomaly detection
    """
    output_dir = create_output_dir()
    
    # Only works for 2D data
    if X_test.shape[1] != 2:
        # If not 2D, use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_test_2d = pca.fit_transform(X_test)
    else:
        X_test_2d = X_test
    
    # 1. Visualize the data points and anomalies
    plt.figure(figsize=(12, 10))
    
    # Plot the scatter plot with colored points
    plt.scatter(X_test_2d[y_test == 0, 0], X_test_2d[y_test == 0, 1], 
                c='blue', label='Normal', alpha=0.6, edgecolors='k', s=50)
    plt.scatter(X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1], 
                c='red', label='Anomaly (True)', alpha=0.6, edgecolors='k', s=100, marker='X')
    
    # Mark the detected anomalies by GMM
    gmm_detected = gmm_scores >= threshold
    normal_detected_as_anomaly = (y_test == 0) & gmm_detected
    plt.scatter(X_test_2d[normal_detected_as_anomaly, 0], X_test_2d[normal_detected_as_anomaly, 1],
                facecolors='none', edgecolors='orange', s=130, label='False Positive', linewidth=2)
    
    # Add contour for decision boundary (only for 2D data)
    if X_test.shape[1] == 2:
        # Create a mesh grid
        x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
        y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Calculate log-likelihood for each point in the grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = -GMMModel.score_samples_static(grid_points, gmm)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contour(xx, yy, Z, levels=[threshold], colors='k', linestyles='--', linewidths=2)
    
    plt.title('Anomaly Detection with GMM', fontsize=15)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gmm_anomaly_detection.png'))
    plt.show()
    
    # 2. Plot ROC curves for comparison
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for GMM
    plt.plot(gmm_metrics['fpr_curve'], gmm_metrics['tpr_curve'], 
             color='darkorange', lw=2, 
             label=f'GMM (AUC = {gmm_metrics["auc_roc"]:.3f})')
    
    # Plot ROC curve for Isolation Forest
    plt.plot(if_metrics['fpr_curve'], if_metrics['tpr_curve'], 
             color='green', lw=2, 
             label=f'Isolation Forest (AUC = {if_metrics["auc_roc"]:.3f})')
    
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_comparison.png'))
    plt.show()
    
    # 3. Plot Precision-Recall curves for comparison
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve for GMM
    plt.plot(gmm_metrics['recall_curve'], gmm_metrics['precision_curve'], 
             color='darkorange', lw=2, 
             label=f'GMM (AP = {gmm_metrics["average_precision"]:.3f})')
    
    # Plot PR curve for Isolation Forest
    plt.plot(if_metrics['recall_curve'], if_metrics['precision_curve'], 
             color='green', lw=2, 
             label=f'Isolation Forest (AP = {if_metrics["average_precision"]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves', fontsize=15)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_comparison.png'))
    plt.show()
    
    # 4. Plot score distributions for normal vs anomaly
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(gmm_scores[y_test == 0], color='blue', label='Normal', kde=True, stat='density', alpha=0.5)
    sns.histplot(gmm_scores[y_test == 1], color='red', label='Anomaly', kde=True, stat='density', alpha=0.5)
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.title('GMM Anomaly Score Distribution')
    plt.xlabel('Anomaly Score (-log likelihood)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Find optimal threshold for IF
    precision, recall, thresholds_pr = precision_recall_curve(y_test, if_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    if_threshold = thresholds_pr[optimal_idx] if len(thresholds_pr) > optimal_idx else thresholds_pr[-1]
    
    sns.histplot(if_scores[y_test == 0], color='blue', label='Normal', kde=True, stat='density', alpha=0.5)
    sns.histplot(if_scores[y_test == 1], color='red', label='Anomaly', kde=True, stat='density', alpha=0.5)
    plt.axvline(x=if_threshold, color='black', linestyle='--', label=f'Threshold: {if_threshold:.3f}')
    plt.title('Isolation Forest Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'))
    plt.show()


def create_real_world_example():
    """
    Create a more realistic anomaly detection scenario using synthetic data.
    
    This example simulates credit card transactions with features like:
    - Transaction amount
    - Hour of day
    - Day of week
    - Distance from home
    - Etc.
    
    A small percentage of transactions are fraudulent (anomalies).
    """
    np.random.seed(42)
    output_dir = create_output_dir()
    
    print("\n--- Creating Realistic Credit Card Fraud Detection Example ---")
    
    # Number of transactions
    n_transactions = 2000
    n_fraudulent = int(n_transactions * 0.03)  # 3% are fraudulent
    n_normal = n_transactions - n_fraudulent
    
    # Generate normal transaction data
    transaction_amount_normal = np.random.lognormal(4, 1, n_normal)  # Most transactions $30-$80
    hour_of_day_normal = np.random.normal(12, 5, n_normal).clip(0, 23)  # Mostly during day
    day_of_week_normal = np.random.randint(0, 7, n_normal)  # Any day of week
    distance_from_home_normal = np.random.exponential(5, n_normal)  # Mostly close to home
    transaction_freq_normal = np.random.poisson(3, n_normal).clip(1, 10)  # Normal frequency
    
    # Generate fraudulent transaction data
    transaction_amount_fraud = np.random.lognormal(6, 2, n_fraudulent)  # Higher amounts
    hour_of_day_fraud = np.random.normal(3, 3, n_fraudulent).clip(0, 23)  # More likely during night
    day_of_week_fraud = np.random.randint(0, 7, n_fraudulent)  # Any day of week
    distance_from_home_fraud = np.random.gamma(10, 5, n_fraudulent)  # Further from home
    transaction_freq_fraud = np.random.poisson(8, n_fraudulent).clip(5, 20)  # Higher frequency
    
    # Combine features
    X_normal = np.column_stack([
        transaction_amount_normal,
        hour_of_day_normal,
        day_of_week_normal,
        distance_from_home_normal,
        transaction_freq_normal
    ])
    
    X_fraud = np.column_stack([
        transaction_amount_fraud,
        hour_of_day_fraud,
        day_of_week_fraud,
        distance_from_home_fraud,
        transaction_freq_fraud
    ])
    
    # Create combined dataset
    X = np.vstack([X_normal, X_fraud])
    y = np.zeros(n_transactions)
    y[n_normal:] = 1  # Label fraudulent transactions as 1
    
    # Feature names for visualization
    feature_names = [
        'Transaction Amount ($)',
        'Hour of Day',
        'Day of Week',
        'Distance from Home (miles)',
        'Transaction Frequency (per week)'
    ]
    
    # Shuffle the data
    idx = np.random.permutation(n_transactions)
    X, y = X[idx], y[idx]
    
    # Create a DataFrame for easier visualization
    df = pd.DataFrame(X, columns=feature_names)
    df['Fraudulent'] = y
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train GMM model
    gmm = GMMModel(n_components=5, covariance_type='full', random_state=42)
    gmm.train(X_train_scaled, scale_data=False)  # Already scaled
    
    # Evaluate model
    metrics, anomaly_scores = evaluate_anomaly_detection(gmm, X_test_scaled, y_test, contamination=0.03)
    
    # Get feature importances through sensitivity analysis
    print("\nFeature Importance Analysis (Sensitivity to Anomalies):")
    feature_importance = []
    baseline_score = -gmm.score_samples(X_test_scaled).mean()
    
    for i in range(X_test_scaled.shape[1]):
        # Copy the data and scramble one feature
        X_scrambled = X_test_scaled.copy()
        X_scrambled[:, i] = np.random.permutation(X_scrambled[:, i])
        
        # Measure the change in log-likelihood
        scrambled_score = -gmm.score_samples(X_scrambled).mean()
        importance = scrambled_score - baseline_score
        feature_importance.append(importance)
        print(f"{feature_names[i]}: {importance:.4f}")
    
    # Normalize feature importance
    feature_importance = np.array(feature_importance)
    feature_importance = feature_importance / feature_importance.sum()
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance for Fraud Detection')
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.show()
    
    # Visualize the data with PCA
    pca = PCA(n_components=2)
    X_test_2d = pca.fit_transform(X_test_scaled)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(X_test_2d[y_test == 0, 0], X_test_2d[y_test == 0, 1], 
                c='blue', label='Normal Transaction', alpha=0.6, s=50)
    plt.scatter(X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1], 
                c='red', label='Fraudulent Transaction', alpha=0.8, s=100, marker='X')
    
    # Mark the detected anomalies by GMM
    gmm_detected = anomaly_scores >= metrics['threshold']
    normal_detected_as_anomaly = (y_test == 0) & gmm_detected
    fraud_not_detected = (y_test == 1) & ~gmm_detected
    
    plt.scatter(X_test_2d[normal_detected_as_anomaly, 0], X_test_2d[normal_detected_as_anomaly, 1],
                facecolors='none', edgecolors='orange', s=130, label='False Positive', linewidth=2)
    plt.scatter(X_test_2d[fraud_not_detected, 0], X_test_2d[fraud_not_detected, 1],
                facecolors='none', edgecolors='purple', s=180, label='False Negative', linewidth=2)
    
    plt.title('Credit Card Fraud Detection with GMM (PCA Visualization)', fontsize=15)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'credit_card_fraud_detection.png'))
    plt.show()
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test, gmm


def main():
    """Main function to run the GMM anomaly detection examples."""
    print("\n=== Gaussian Mixture Model for Anomaly Detection ===\n")
    print("This script demonstrates the use of GMM for anomaly detection.")
    
    # Part 1: Simple 2D example with visualization
    print("\n--- Part 1: 2D Anomaly Detection Example ---")
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=1000, n_features=2, contamination=0.05)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train a GMM model
    gmm = train_gmm_model(X_train, n_components=3)
    
    # Evaluate anomaly detection performance
    metrics, anomaly_scores = evaluate_anomaly_detection(gmm, X_test, y_test)
    
    # Compare with Isolation Forest
    if_metrics, if_scores = compare_with_isolation_forest(X_train, X_test, y_test)
    
    # Visualize results
    visualize_anomaly_detection(X_test, y_test, anomaly_scores, if_scores, 
                               metrics, if_metrics, metrics['threshold'])
    
    # Part 2: Realistic example with credit card fraud detection
    print("\n--- Part 2: Credit Card Fraud Detection Example ---")
    df, X_train, X_test, y_train, y_test, gmm = create_real_world_example()
    
    # Show the density plot for each component
    gmm.plot_density(X_test[:100], feature_idx=(0, 3))  # Amount vs Distance
    
    print("\nGMM successfully trained and evaluated for anomaly detection!")
    
    # Save the trained model
    output_dir = create_output_dir()
    model_path = os.path.join(output_dir, 'gmm_anomaly_detection.pkl')
    gmm.save_model(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main() 