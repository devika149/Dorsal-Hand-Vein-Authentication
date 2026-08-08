import tensorflow as tf
import os

model_path = r"e:\Srishti\Student project haritha\Dorsal Vein\models\hand_detection_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    print("Output shape:", model.output_shape)
    print("Output activation:", model.layers[-1].activation.__name__ if hasattr(model.layers[-1], 'activation') else "N/A")
else:
    print("Model not found")
