import cv2
import numpy as np
from keras.models import load_model

class SignLanguageDetector:
    def __init__(self, model_path, labels):
        self.model_path = model_path
        self.labels = labels
        self.model = load_model(model_path)
        self.class_labels = {idx: label for idx, label in enumerate(labels)}
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (64, 64))
        frame = frame.astype('float32')
        frame /= 255
        return frame.reshape(1, 64, 64, 3)
    def predict_gesture(self, frame):
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(processed_frame)
        class_index = np.argmax(prediction)
        return self.class_labels[class_index]
    def start_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Could not open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Predict the sign language
            sign_language = self.predict_gesture(frame)
            # Display the prediction
            cv2.putText(frame, sign_language, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Sign Language Detection', frame)
            # Break the loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the webcam
        cap.release()
        cv2.destroyAllWindows()

# Usage
labels = ['Hello', 'Yes', 'No', 'ThankYou', 'ILikeYou']
detector = SignLanguageDetector('sign_language_model.h5', labels)
detector.start_webcam()
