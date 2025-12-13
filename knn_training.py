"""
KNN Training Script with Dimensionality Reduction
Optimized for better accuracy with reduced features
"""

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# File paths
X_filepath = "extracted_features/X.npy"
y_filepath = "extracted_features/y.npy"
model_filepath = "knn_model.pkl"


# ============================================================================
# KNN HYPERPARAMETERS - EXPERIMENT WITH THESE VALUES
# ============================================================================

"""
PARAMETER TUNING GUIDE FOR KNN:

1. n_neighbors (k): Number of nearest neighbors to consider
    Values to try: [3, 5, 7, 9, 11, 15, 21, 25]
    - Lower k (3-5): More sensitive to noise, might overfit
    - Medium k (7-11): Usually best balance
    - Higher k (15-25): Smoother boundaries, might underfit
    Start with: 7 or 9

2. weights: How to weight the neighbors
    Values to try: ['uniform', 'distance']
    - 'uniform': All neighbors weighted equally
    - 'distance': Closer neighbors have more influence (RECOMMENDED)
    Start with: 'distance'

3. metric: Distance metric to use
    Values to try: ['euclidean', 'manhattan', 'cosine', 'minkowski']
    - 'euclidean': Standard distance (default)
    - 'manhattan': City-block distance (good for high dims)
    - 'cosine': Angle-based similarity (good for normalized data)
    - 'minkowski': Generalized distance (with p parameter)
    Start with: 'euclidean' or 'manhattan'

4. p (only for minkowski): Power parameter
    Values to try: [1, 2, 3]
    - p=1: Manhattan distance
    - p=2: Euclidean distance
    - p=3+: Higher order distances
    Start with: 2

5. algorithm: Algorithm to compute nearest neighbors
    Values to try: ['auto', 'ball_tree', 'kd_tree', 'brute']
    - 'auto': Let sklearn choose (RECOMMENDED)
    - 'ball_tree': Good for high dimensions
    - 'kd_tree': Fast for low dimensions
    - 'brute': Slow but accurate
    Start with: 'auto'

RECOMMENDED STARTING CONFIGURATIONS:
Config 1: k=7, weights='distance', metric='euclidean'
Config 2: k=9, weights='distance', metric='manhattan'
Config 3: k=11, weights='distance', metric='euclidean'
Config 4: k=7, weights='uniform', metric='euclidean'
"""

# Set your parameters here: (MAX: 69.99% accuracy, 71.65% with 8554 images)
KNN_PARAMS = {
    'n_neighbors': 7,           # Change this: [3, 5, *7, 9, 11, 15, 21]
    'weights': 'distance',       # Change this: ['uniform', '*distance']
    'metric': 'cosine',       # Change this: ['euclidean', 'manhattan', '*cosine']
    'algorithm': 'auto',         # Usually keep as 'auto'
    'n_jobs': -1                 # Use all CPU cores
}

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
    plt.savefig('knn_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to knn_visualization.png")


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
        knn = joblib.load(model_filepath)
        print("Model loaded.")
    else:
        print(f"\nTraining new KNN model with parameters:")
        for key, val in KNN_PARAMS.items():
            print(f"  {key}: {val}")
        
        knn = KNeighborsClassifier(**KNN_PARAMS)
        knn.fit(X_train, y_train)
        
        joblib.dump(knn, model_filepath)
        print(f"Model saved to {model_filepath}")
    
    # Predictions
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Report
    print("\n" + "="*50)
    print("ACCURACY REPORT")
    print("="*50)
    print(f"Model: K-Nearest Neighbors")
    print(f"Parameters: {KNN_PARAMS}")
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