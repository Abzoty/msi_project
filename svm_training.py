"""
SVM Training Script with Dimensionality Reduction
Optimized for better accuracy with reduced features
"""

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# File paths
X_filepath = "extracted_features/features.npy"
y_filepath = "extracted_features/labels.npy"
model_filepath = "svm_model.pkl"


# ============================================================================
# SVM HYPERPARAMETERS - EXPERIMENT WITH THESE VALUES
# ============================================================================

"""
PARAMETER TUNING GUIDE FOR SVM:

1. kernel: Type of kernel function
    Values to try: ['rbf', 'linear', 'poly', 'sigmoid']
    - 'rbf': Radial Basis Function (RECOMMENDED, works well for most cases)
    - 'linear': Linear kernel (fast, good for linearly separable data)
    - 'poly': Polynomial kernel (can model complex boundaries)
    - 'sigmoid': Sigmoid kernel (similar to neural networks)
    Start with: 'rbf'

2. C: Regularization parameter (controls trade-off between margin and misclassification)
    Values to try: [0.1, 1, 10, 50, 100, 200]
    - Low C (0.1-1): Soft margin, more regularization, may underfit
    - Medium C (10-50): Good balance (RECOMMENDED)
    - High C (100-200): Hard margin, less regularization, may overfit
    Start with: 10 or 50

3. gamma: Kernel coefficient (only for 'rbf', 'poly', 'sigmoid')
    Values to try: ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    - 'scale': 1 / (n_features * X.var()) - RECOMMENDED
    - 'auto': 1 / n_features
    - Low gamma (0.001-0.01): Smooth decision boundary
    - High gamma (0.1-1): Complex decision boundary, may overfit
    Start with: 'scale'

4. degree: Degree of polynomial (only for 'poly' kernel)
    Values to try: [2, 3, 4, 5]
    - degree=2: Quadratic
    - degree=3: Cubic (RECOMMENDED for poly)
    - degree=4+: Higher complexity
    Start with: 3

5. class_weight: Weights for imbalanced classes
    Values to try: [None, 'balanced']
    - None: All classes have equal weight
    - 'balanced': Automatically adjust weights inversely proportional to class frequencies
    Start with: None (your dataset is balanced)

RECOMMENDED STARTING CONFIGURATIONS:
Config 1: kernel='rbf', C=10, gamma='scale'
Config 2: kernel='rbf', C=50, gamma='scale'
Config 3: kernel='rbf', C=100, gamma='scale'
Config 4: kernel='linear', C=1
Config 5: kernel='poly', C=10, degree=3, gamma='scale'
"""

# Set your parameters here: (MAX: 77.71% accuracy, 76.62% with 8554 images)
SVM_PARAMS = {
    'kernel': 'rbf',        # Change this: ['*rbf', 'linear', 'poly', 'sigmoid']
    'C': 10,              # Change this: [0.1, 1, *10, 50, 100, 200]
    'gamma': 'scale',       # Change this: ['*scale', 'auto', 0.001, 0.01, 0.1, 1]
    'random_state': 42
}

# For 'poly' kernel, add this:
# 'degree': 3

# ============================================================================


def read_data():
    """Load features and labels."""
    try:
        X = np.load(X_filepath)
        y = np.load(y_filepath)
        print(f"Loaded data: X shape={X.shape}, y shape={y.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def plot_visualizations(X, y, X_train, y_train, X_test, y_test, y_pred):
    """Create comprehensive visualizations."""
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Class Distribution
    ax1 = plt.subplot(2, 3, 1)
    unique, counts = np.unique(y, return_counts=True)
    ax1.bar(range(len(unique)), counts, color=[color_map[cls] for cls in unique])
    ax1.set_xlabel('Class Label')
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    ax1.set_xticks(range(len(unique)))
    ax1.set_xticklabels(unique)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. PCA 2D - All Data
    ax2 = plt.subplot(2, 3, 2)
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X)
    
    for cls in unique_classes:
        mask = y == cls
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[color_map[cls]], label=cls, alpha=0.6, s=20)
    
    ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('PCA 2D - All Data')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. PCA 3D
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)
    
    for cls in unique_classes:
        mask = y == cls
        ax3.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
                   c=[color_map[cls]], label=cls, alpha=0.6, s=15)
    
    ax3.set_title('PCA 3D')
    ax3.legend()
    
    # 4. Train/Test Split
    ax4 = plt.subplot(2, 3, 4)
    X_train_pca = pca_2d.transform(X_train)
    X_test_pca = pca_2d.transform(X_test)
    
    for cls in unique_classes:
        train_mask = y_train == cls
        test_mask = y_test == cls
        
        if np.any(train_mask):
            ax4.scatter(X_train_pca[train_mask, 0], X_train_pca[train_mask, 1],
                       c=[color_map[cls]], marker='o', alpha=0.5, s=30, label=f'{cls} (train)')
        
        if np.any(test_mask):
            ax4.scatter(X_test_pca[test_mask, 0], X_test_pca[test_mask, 1],
                       c=[color_map[cls]], marker='s', alpha=0.8, s=50,
                       edgecolors='red', linewidth=1.5)
    
    ax4.set_title('Train/Test Split')
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)
    
    # 5. Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes, ax=ax5)
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')
    ax5.set_title('Confusion Matrix')
    
    # 6. Predictions
    ax6 = plt.subplot(2, 3, 6)
    X_test_pca = pca_2d.transform(X_test)
    correct = y_test == y_pred
    incorrect = ~correct
    
    for cls in unique_classes:
        cls_correct = correct & (y_test == cls)
        if np.any(cls_correct):
            ax6.scatter(X_test_pca[cls_correct, 0], X_test_pca[cls_correct, 1],
                       c=[color_map[cls]], marker='o', alpha=0.7, s=50,
                       edgecolors='green', linewidth=2, label=f'{cls} ✓')
    
    if np.any(incorrect):
        ax6.scatter(X_test_pca[incorrect, 0], X_test_pca[incorrect, 1],
                   c='red', marker='x', s=150, linewidth=3, label='Misclassified')
    
    ax6.set_title('Predictions (X = misclassified)')
    ax6.legend(fontsize=7)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svm_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to svm_visualization.png")


def main():
    # Load data
    X, y = read_data()
    
    # Train/test split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Train or load model
    if os.path.isfile(model_filepath):
        print(f"\nLoading model from {model_filepath}")
        svm = joblib.load(model_filepath)
        print("Model loaded.")
    else:
        print(f"\nTraining new SVM model with parameters:")
        for key, val in SVM_PARAMS.items():
            print(f"  {key}: {val}")
        
        svm = SVC(**SVM_PARAMS)
        svm.fit(X_train, y_train)
        
        joblib.dump(svm, model_filepath)
        print(f"Model saved to {model_filepath}")
    
    # Predictions
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Report
    print("\n" + "="*50)
    print("ACCURACY REPORT")
    print("="*50)
    print(f"Model: Support Vector Machine (SVM)")
    print(f"Parameters: {SVM_PARAMS}")
    print(f"Features: {X.shape[1]} (PCA reduced)")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("-"*50)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("="*50)
    
    # Per-class accuracy with class names
    class_names = {0: 'glass', 1: 'paper', 2: 'cardboard', 3: 'plastic', 4: 'metal', 5: 'trash'}
    cm = confusion_matrix(y_test, y_pred)
    print("\nPer-class accuracy:")
    for i, cls in enumerate(np.unique(y)):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            print(f"  Class {cls} ({class_names[cls]:12}): {acc:.1f}%")
    
    # Generate outputs
    plot_visualizations(X, y, X_train, y_train, X_test, y_test, y_pred)
    
    print("\nDone!")


if __name__ == "__main__":
    main()