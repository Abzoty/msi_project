from HOG_ColorHistogramsFeatureExtraction import extract_features
import numpy as np
import joblib
import cv2


knn_model_filepath = "knn_model.pkl"
knn = joblib.load(knn_model_filepath)

svm_model_filepath = "svm_model.pkl"
svm = joblib.load(svm_model_filepath)


class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
            
def process_frame(frame, model):
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    features = extract_features(image_rgb)
    X = features.reshape(1, -1)
    prediction_index = model.predict(X).item()
    return class_names[prediction_index]            
            
def main():

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        predicted_class_knn = process_frame(frame, knn)
        predicted_class_svm = process_frame(frame, svm)
        print(f"Predicted class (KNN): {predicted_class_knn} --------- Predicted class (SVM): {predicted_class_svm}\n")

        # Display the resulting frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Predicted class (KNN): {predicted_class_knn}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Predicted class (SVM): {predicted_class_svm}", (10, 60), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


