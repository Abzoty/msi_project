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
X_filepath = "extracted_features/X.npy"
y_filepath = "extracted_features/y.npy"
class_map_filepath = "extracted_features/class_map.txt"
model_filepath = "svm_model.pkl"

# ============================================================================
# SVM HYPERPARAMETERS
# ============================================================================

# Set your parameters here: (MAX: 88.54% accuracy)
SVM_PARAMS = {
    'kernel': 'rbf',        # Change this: ['*rbf', 'linear', 'poly', 'sigmoid']
    'C': 10,                # Change this: [0.1, 1, *10, 50, 100, 200]
    'gamma': 'scale',       # Change this: ['*scale', 'auto', 0.001, 0.01, 0.1, 1]
    'random_state': 42,
    'probability': False
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

def load_class_map():
    """Reads the class mapping from the text file."""
    if not os.path.exists(class_map_filepath):
        print(f"⚠️ Warning: {class_map_filepath} not found. Using numeric labels.")
        return {}
    
    mapping = {}
    with open(class_map_filepath, "r") as f:
        for line in f:
            if ":" in line:
                parts = line.strip().split(":")
                idx = int(parts[0].strip())
                name = parts[1].strip()
                mapping[idx] = name
    return mapping

def plot_visualizations(X, y, X_train, y_train, X_test, y_test, y_pred, class_map):
    """Create comprehensive visualizations."""
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    # Helper to get name
    def get_name(cls_idx):
        return class_map.get(cls_idx, str(cls_idx))

    fig = plt.figure(figsize=(18, 12))
    
    # 1. Class Distribution
    ax1 = plt.subplot(2, 3, 1)
    unique, counts = np.unique(y, return_counts=True)
    labels = [get_name(cls) for cls in unique]
    ax1.bar(range(len(unique)), counts, color=[color_map[cls] for cls in unique])
    ax1.set_xlabel('Class Label')
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    ax1.set_xticks(range(len(unique)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. PCA 2D - All Data
    ax2 = plt.subplot(2, 3, 2)
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X)
    
    for cls in unique_classes:
        mask = y == cls
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                    c=[color_map[cls]], label=get_name(cls), alpha=0.6, s=20)
    
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
                    c=[color_map[cls]], label=get_name(cls), alpha=0.6, s=15)
        
    ax3.set_title('PCA 3D')
    
    # 4. Train/Test Split
    ax4 = plt.subplot(2, 3, 4)
    X_train_pca = pca_2d.transform(X_train)
    X_test_pca = pca_2d.transform(X_test)
    
    for cls in unique_classes:
        train_mask = y_train == cls
        test_mask = y_test == cls
        
        if np.any(train_mask):
            ax4.scatter(X_train_pca[train_mask, 0], X_train_pca[train_mask, 1],
                        c=[color_map[cls]], marker='o', alpha=0.5, s=30)
        
        if np.any(test_mask):
            ax4.scatter(X_test_pca[test_mask, 0], X_test_pca[test_mask, 1],
                        c=[color_map[cls]], marker='s', alpha=0.8, s=50,
                        edgecolors='red', linewidth=1.5)
    
    ax4.set_title('Train (o) vs Test (s)')
    ax4.grid(alpha=0.3)
    
    # 5. Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax5)
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
                        edgecolors='green', linewidth=2, label=f'{get_name(cls)} ✓')
    
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
    # Load data and class map
    X, y = read_data()
    class_map = load_class_map()
    
    if not class_map:
        # Fallback if file is missing
        class_map = {i: str(i) for i in np.unique(y)}

    # Train/test split (Stratified to ensure 80/20 split PER CLASS)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    # Validate Distribution
    print("\n" + "="*50)
    print("DATA SPLIT VERIFICATION (80/20 per class)")
    print("="*50)
    print(f"{'Class Name':<15} | {'Total':<6} | {'Train':<6} | {'Test':<6}")
    print("-" * 45)
    
    for cls in np.unique(y):
        n_total = np.sum(y == cls)
        n_train = np.sum(y_train == cls)
        n_test = np.sum(y_test == cls)
        cls_name = class_map.get(cls, str(cls))
        print(f"{cls_name:<15} | {n_total:<6} | {n_train:<6} | {n_test:<6}")
    print("-" * 45)
    print(f"Total samples   | {len(y):<6} | {len(y_train):<6} | {len(y_test):<6}")
    print("="*50)

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
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("="*50)
    
    # Per-class accuracy using dynamic map
    cm = confusion_matrix(y_test, y_pred)
    print("\nPer-class accuracy:")
    for i, cls in enumerate(np.unique(y)):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            name = class_map.get(cls, f"Class {cls}")
            print(f"  {name:15}: {acc:.1f}%")
    
    # Generate outputs
    plot_visualizations(X, y, X_train, y_train, X_test, y_test, y_pred, class_map)
    
    print("\nDone!")

if __name__ == "__main__":
    main()