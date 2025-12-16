from feature_extraction import extract_features
import numpy as np
import joblib
import cv2

# --- 1. Load Models Correctly ---
knn_model_filepath = "knn_model.pkl"
knn = joblib.load(knn_model_filepath)

# FIX 1: Load the actual BoVW model, not the KNN model again
bovw_model_filepath = "extracted_features/bovw.pkl"
bovw = joblib.load(bovw_model_filepath)

svm_model_filepath = "svm_model.pkl"
svm = joblib.load(svm_model_filepath)

scaler_filepath = "extracted_features/scaler.pkl"
scaler = joblib.load(scaler_filepath)

# FIX 2: Load the PCA model (required to match the 200 features expected by KNN/SVM)
pca_filepath = "extracted_features/pca.pkl"
pca = joblib.load(pca_filepath)

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]

def process_frame(frame, model):
    # Global variables for the transformers
    global scaler, bovw, pca

    # Preprocessing (must match training exactly)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Note: Resize is handled inside extract_features, but doing it here doesn't hurt
    # image_rgb = cv2.resize(image_rgb, (128, 128)) 
    
    # 1. Extract Raw Features (HOG + LBP + GLCM + Color + BoVW)
    features = extract_features(image_rgb, bovw)
    
    # Reshape for sklearn (1 sample, N features)
    X = features.reshape(1, -1)
    
    # FIX 3: Use .transform(), NEVER .fit_transform() during inference
    X_Scaled = scaler.transform(X)
    
    # FIX 4: Apply PCA to reduce dimensions from ~8000 to 200
    X_PCA = pca.transform(X_Scaled)
    
    # Predict
    probs = model.predict_proba(X_PCA)[0]
    print(probs)
    max_prob = probs.max()
    pred_index = probs.argmax()
    
    # Threshold for "Unknown"
    if max_prob < 0.6:
        return class_names[-1] 
        
    return class_names[pred_index]

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Run prediction
        try:
            predicted_class_knn = process_frame(frame, knn)
            predicted_class_svm = process_frame(frame, svm)
            
            # Console Output
            print(f"KNN: {predicted_class_knn} | SVM: {predicted_class_svm}")

            # Display on Frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"KNN: {predicted_class_knn}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"SVM: {predicted_class_svm}", (10, 70), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Recycle Smart', frame)

        except Exception as e:
            print(f"Error processing frame: {e}")
            break
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # frame = cv2.imread("images/trash/109.jpg")
    # predicted_class_knn = process_frame(frame, knn)
    # predicted_class_svm = process_frame(frame, svm)
    
    # # Console Output
    # print(f"KNN: {predicted_class_knn} | SVM: {predicted_class_svm}")