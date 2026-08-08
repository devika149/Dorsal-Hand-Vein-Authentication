import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    data_dir = os.path.join("data", "Processed")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found. Run preprocess.py first.")
        return

    # Image parameters
    img_size = (224, 224)
    batch_size = 32

    # Feature Extractor: VGG16 pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Add a global average pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    
    # Save the feature extractor
    feature_extractor.save(os.path.join("models", "cnn_model.h5"))
    print("CNN feature extractor saved to models/cnn_model.h5")

    # Load images and extract features
    images = []
    labels = []
    
    # Load and augment data
    images = []
    labels = []
    
    user_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"Loading data from {len(user_folders)} users with augmentation...")
    
    # Simple augmentation for robust features
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        fill_mode='nearest'
    )
    
    for user_folder in user_folders:
        user_path = os.path.join(data_dir, user_folder)
        image_files = os.listdir(user_path)
        for img_name in image_files:
            img_path = os.path.join(user_path, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Add original image
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch.copy())
            features = feature_extractor.predict(img_preprocessed, verbose=0)
            images.append(features.flatten())
            labels.append(user_folder)
            
            # Add augmented versions
            it = datagen.flow(img_batch, batch_size=1)
            for _ in range(10): # Increased to 10 for better coverage
                img_aug = next(it)[0]
                img_aug_preprocessed = tf.keras.applications.vgg16.preprocess_input(np.expand_dims(img_aug, axis=0))
                features = feature_extractor.predict(img_aug_preprocessed, verbose=0)
                images.append(features.flatten())
                labels.append(user_folder)

    X = np.array(images)
    y = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Standardize features - CRITICAL for SVM confidence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save label encoder and scaler for consistency
    with open(os.path.join("models", "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join("models", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Label encoder and Scaler saved to models/")

    # Train SVM with RBF kernel - better for non-linear decision boundaries
    X_train, y_train = X_scaled, y_encoded
    print(f"Training SVM classifier on {len(X_train)} samples for {len(le.classes_)} classes...")
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
    svm.fit(X_train, y_train)

    # Evaluate on training data
    y_pred = svm.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"Training Recognition Accuracy: {accuracy * 100:.2f}%")

    # Save SVM model
    with open(os.path.join("models", "Svm_model.pkl"), "wb") as f:
        pickle.dump(svm, f)
    print("SVM model saved to models/Svm_model.pkl")

if __name__ == "__main__":
    main()
