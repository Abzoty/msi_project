from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import numpy as np
import os 
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Define file paths
X_Scaled_filepath = "extracted_features/features.npy"
labels_y_filepath = "extracted_features/labels.npy"
knn_model_filepath = "knn_model.pkl"
PREDICTION_OUTPUT_FILE = "prediction_report.txt"

def read_data():
    """Loads scaled features, labels, and the scaler object."""
    try:
        X = np.load(X_Scaled_filepath)
        y = np.load(labels_y_filepath)
        return  X, y
    except Exception as e:
        print(f"Error loading data files: {e}. Ensure features_X.npy, labels_y.npy, and scaler.pkl exist.")
        raise 
    
    
def write_prediction_report(y_test, y_pred):
    """
    Writes a side-by-side comparison of true vs. predicted labels to a text file.
    """
    print(f"\nWriting detailed predictions to {PREDICTION_OUTPUT_FILE}...")
    
    try:
        with open(PREDICTION_OUTPUT_FILE, 'w') as f:
            
            # Write Header
            f.write("=" * 60 + "\n")
            f.write(f"PREDICTION REPORT FOR KNN MODEL\n")
            f.write(f"Test Set Size: {len(y_test)} samples\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Sample Index':<15} | {'Actual Label (True)':<20} | {'Predicted Label':<20}\n")
            f.write("-" * 60 + "\n")
            
            # Write Data Rows
            for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
                status = " "
                if actual != predicted:
                    status = "MISCLASSIFIED" 
                
                f.write(f"{i:<15} | {actual:<20} | {predicted:<20} ({status})\n")
        
        print(f"Prediction list written successfully to {PREDICTION_OUTPUT_FILE}")
    
    except IOError as e:
        print(f"ERROR: Could not write prediction file: {e}")    


def plot_data_visualization(X, y, X_train, y_train, X_test, y_test, y_pred):
    """
    Creates comprehensive visualizations of the data and model performance.
    """
    # Get unique classes and assign distinct colors
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Class Distribution (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    unique, counts = np.unique(y, return_counts=True)
    bars = ax1.bar(range(len(unique)), counts, color=[color_map[cls] for cls in unique])
    ax1.set_xlabel('Class Label', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title('Class Distribution in Dataset', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(unique)))
    ax1.set_xticklabels(unique, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. PCA 2D Visualization - Full Dataset (Top Center)
    ax2 = plt.subplot(2, 3, 2)
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)
    
    for cls in unique_classes:
        mask = y == cls
        ax2.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                   c=[color_map[cls]], label=cls, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    
    ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} var)', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} var)', fontsize=11)
    ax2.set_title('PCA 2D Visualization - All Data', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # 3. PCA 3D Visualization (Top Right)
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)
    
    for cls in unique_classes:
        mask = y == cls
        ax3.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
                   c=[color_map[cls]], label=cls, alpha=0.6, s=20, edgecolors='k', linewidth=0.5)
    
    ax3.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})', fontsize=9)
    ax3.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})', fontsize=9)
    ax3.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})', fontsize=9)
    ax3.set_title('PCA 3D Visualization', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7)
    
    # 4. Train vs Test Split Visualization (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    X_train_pca = pca_2d.transform(X_train)
    X_test_pca = pca_2d.transform(X_test)
    
    for cls in unique_classes:
        train_mask = y_train == cls
        test_mask = y_test == cls
        
        if np.any(train_mask):
            ax4.scatter(X_train_pca[train_mask, 0], X_train_pca[train_mask, 1],
                       c=[color_map[cls]], marker='o', alpha=0.6, s=40, 
                       edgecolors='k', linewidth=0.5, label=f'{cls} (train)')
        
        if np.any(test_mask):
            ax4.scatter(X_test_pca[test_mask, 0], X_test_pca[test_mask, 1],
                       c=[color_map[cls]], marker='s', alpha=0.8, s=60,
                       edgecolors='red', linewidth=1.5, label=f'{cls} (test)')
    
    ax4.set_xlabel('PC1', fontsize=11)
    ax4.set_ylabel('PC2', fontsize=11)
    ax4.set_title('Train/Test Split (circles=train, squares=test)', fontsize=12, fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax4.grid(alpha=0.3)
    
    # 5. Confusion Matrix (Bottom Center)
    ax5 = plt.subplot(2, 3, 5)
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes,
                cbar_kws={'label': 'Count'}, ax=ax5)
    
    ax5.set_xlabel('Predicted Label', fontsize=11)
    ax5.set_ylabel('True Label', fontsize=11)
    ax5.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # 6. Prediction Results Visualization (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    X_test_pca = pca_2d.transform(X_test)
    
    # Plot correct predictions
    correct_mask = y_test == y_pred
    incorrect_mask = ~correct_mask
    
    for cls in unique_classes:
        cls_correct = correct_mask & (y_test == cls)
        if np.any(cls_correct):
            ax6.scatter(X_test_pca[cls_correct, 0], X_test_pca[cls_correct, 1],
                       c=[color_map[cls]], marker='o', alpha=0.7, s=60,
                       edgecolors='green', linewidth=2, label=f'{cls} ✓')
    
    # Plot incorrect predictions with X markers
    if np.any(incorrect_mask):
        ax6.scatter(X_test_pca[incorrect_mask, 0], X_test_pca[incorrect_mask, 1],
                   c='red', marker='x', s=200, linewidth=3, label='Misclassified', zorder=5)
    
    ax6.set_xlabel('PC1', fontsize=11)
    ax6.set_ylabel('PC2', fontsize=11)
    ax6.set_title('Prediction Results (X = misclassified)', fontsize=12, fontweight='bold')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('knn_data_visualization.png', dpi=300, bbox_inches='tight')



class KNNClassifier:
    """Wrapper class for Scikit-learn's KNeighborsClassifier."""
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        # Initialize with the base Scikit-learn classifier
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    
    def save(self, filepath):
        """Saves the *underlying* Scikit-learn model."""
        joblib.dump(self.classifier, filepath)
        print(f"Model successfully saved to {filepath}")

    def load(self, filepath):
        """Loads the Scikit-learn model and assigns it to self.classifier."""
        self.classifier = joblib.load(filepath)
        return self
    
def main():
    
    X, y = read_data()
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in split.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    

    knn = KNNClassifier(3)
    model_was_loaded = False
    
    if os.path.isfile(knn_model_filepath):
        try:
            print(f"model file exists, attempting to load model from {knn_model_filepath}")
            knn.load(knn_model_filepath)
            model_was_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"An unexpected error occurred during loading: {e}. Training new model.")

    
    if not model_was_loaded:
        print(f"Model not found or load failed. Training a new model.")
        knn.fit(X_train, y_train)
        knn.save(knn_model_filepath)

    
    # --- ACCURACY REPORT ---
    y_predict = knn.predict(X_test)
    write_prediction_report(y_test, y_predict)
    
    
    accuracy = accuracy_score(y_predict, y_test)
    
    print("\n" + "="*30)
    print(f"       ACCURACY REPORT      ")
    print("="*30)
    print(f"Model Type: K-Nearest Neighbors (k={knn.n_neighbors})")
    print(f"Total Features: {X.shape[1]}")
    print(f"Dataset Size (Total): {X.shape[0]} samples")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")
    print("-" * 30)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("="*30)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_data_visualization(X, y, X_train, y_train, X_test, y_test, y_predict)
    
    print("Done")


if __name__ == "__main__":
    main()