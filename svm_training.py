from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import os 
from sklearn.model_selection import StratifiedShuffleSplit



# Define file paths
X_Scaled_filepath = "extracted_features/features_X.npy"
labels_y_filepath = "extracted_features/labels_y.npy"
scaler_filepath = "extracted_features/scaler.pkl"
svm_model_filepath = "svm_model.pkl"
PREDICTION_OUTPUT_FILE = "prediction_report_svm.txt"

def read_data():
    """Loads scaled features, labels, and the scaler object."""
    try:
        Scaler = joblib.load(scaler_filepath)
        X = np.load(X_Scaled_filepath)
        y = np.load(labels_y_filepath)
        return Scaler, X, y
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
            f.write(f"PREDICTION REPORT FOR SVM MODEL\n")
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


class SVMClassifier:
    """Wrapper class for Scikit-learn's Support Vector Machine Classifier."""
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        # Initialize with the base Scikit-learn SVM classifier
        self.classifier = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)

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
    
    Scaler , X , y = read_data()
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in split.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    

    svm = SVMClassifier(kernel='rbf', C=60.0, gamma= 'scale')
    model_was_loaded = False
    
    if os.path.isfile(svm_model_filepath):
        try:
            print(f"model file exists, attempting to load model from {svm_model_filepath}")
            svm.load(svm_model_filepath)
            model_was_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"An unexpected error occurred during loading: {e}. Training new model.")

    
    if not model_was_loaded:
        print(f"Model not found or load failed. Training a new model.")
        svm.fit(X_train, y_train)
        svm.save(svm_model_filepath)

    
    # --- ACCURACY REPORT ---
    y_predict = svm.predict(X_test)
    write_prediction_report(y_test, y_predict)
    
    
    accuracy = accuracy_score(y_test, y_predict)
    
    print("\n" + "="*30)
    print(f"       ACCURACY REPORT      ")
    print("="*30)
    print(f"Model Type: Support Vector Machine (SVM)")
    print(f"Kernel: {svm.kernel}")
    print(f"C Parameter: {svm.C}")
    print(f"Gamma: {svm.gamma}")
    print(f"Total Features: {X.shape[1]}")
    print(f"Dataset Size (Total): {X.shape[0]} samples")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")
    print("-" * 30)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("="*30)
    print("Done")


if __name__ == "__main__":
    main()