'''Adding a description for this project'''


import tensorflow as tf
import numpy as np
import pickle
import os
import cv2
from sklearn.preprocessing import LabelEncoder

def main():
    # Load assets
    cnn = tf.keras.models.load_model(os.path.join("models", "cnn_model.h5"))
    with open(os.path.join("models", "Svm_model.pkl"), "rb") as f:
        svm = pickle.load(f)
    
    with open(os.path.join("models", "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    
    classes = le.classes_
    # Test on one image
    test_user = classes[0]
    test_user_dir = os.path.join("data", "Processed", test_user)
    test_img_name = os.listdir(test_user_dir)[0]
    test_img_path = os.path.join(test_user_dir, test_img_name)
    
    img = cv2.imread(test_img_path)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array.astype(np.float32))
    
    features = cnn.predict(img_array)
    probs = svm.predict_proba(features)
    pred_idx = np.argmax(probs[0])
    pred_label = le.classes_[pred_idx]
    
    print(f"Test on {test_img_path}")
    print(f"Actual: {test_user}, Predicted: {pred_label}, Confidence: {probs[0][pred_idx]:.4f}")

if __name__ == "__main__":
    main()
