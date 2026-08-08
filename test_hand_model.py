import tensorflow as tf
import numpy as np
from PIL import Image
import os

model_path = r"models\hand_detection_model.h5"
hand_dir = r"data\han non hand\hand"
non_hand_dir = r"data\han non hand\non hand"

model = tf.keras.models.load_model(model_path)

results = []

def test_on_dir(directory, name):
    results.append(f"\n--- Testing on {name} ---")
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))][:5]
    for f in files:
        img = Image.open(f).convert('RGB').resize((224, 224))
        img_np = np.array(img).astype('float32')
        
        # Method 1: [0, 1] scaling
        input1 = img_np / 255.0
        p1 = model.predict(np.expand_dims(input1, 0), verbose=0)[0][0]
        
        # Method 2: VGG16 Preprocessing
        input2 = tf.keras.applications.vgg16.preprocess_input(img_np.copy())
        p2 = model.predict(np.expand_dims(input2, 0), verbose=0)[0][0]
        
        results.append(f"File: {os.path.basename(f)} | [0,1]: {p1:.4f} | VGG16: {p2:.4f}")

test_on_dir(hand_dir, "HANDS")
test_on_dir(non_hand_dir, "NON-HANDS")

with open(r"e:\Srishti\Student project haritha\Dorsal Vein\model_test_results.txt", "w") as f:
    f.write("\n".join(results))
