import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("fruit_vegetable_model.keras")

# Dataset path used during training
train_dir = "dataset/Training/Training"

# Get class names
class_names = sorted([
    folder for folder in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, folder))
])

# Folder containing test images
test_folder = "test_images"

print("\n==============================")
print("FRUIT & VEGETABLE PREDICTION")
print("==============================\n")

# Process every image
for filename in os.listdir(test_folder):

    image_path = os.path.join(test_folder, filename)

    # Skip non-image files
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize((100, 100))

    # Convert to array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    # Get predicted class
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]

    # Confidence
    confidence = predictions[0][predicted_index] * 100

    print("Image      :", filename)
    print("Prediction :", predicted_class)
    print("Confidence :", f"{confidence:.2f}%")
    print("------------------------------")

print("\nPrediction completed!")