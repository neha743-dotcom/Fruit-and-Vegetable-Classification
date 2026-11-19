import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Dataset paths
TRAIN_DIR = "dataset/Training/Training"
TEST_DIR = "dataset/Test"

# Settings
IMAGE_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 5

# Model file
MODEL_PATH = "fruit_vegetable_model.keras"

print("=" * 50)
print("FRUIT AND VEGETABLE CLASSIFICATION")
print("=" * 50)

# Check folders
if not os.path.exists(TRAIN_DIR):
    print("Training folder not found:")
    print(TRAIN_DIR)
    exit()

if not os.path.exists(TEST_DIR):
    print("Test folder not found:")
    print(TEST_DIR)
    exit()

print("Training folder found")
print("Test folder found")

# Load training dataset
print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load test dataset
print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get class names
class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nNumber of classes:", num_classes)

print("\nFirst 10 classes:")

for name in class_names[:10]:
    print("-", name)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# Create CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(100, 100, 3)
    ),

    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.5
    ),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model created successfully.")

# Train model
print("\n" + "=" * 50)
print("STARTING TRAINING")
print("=" * 50)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

# Save model
model.save(MODEL_PATH)

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED")
print("=" * 50)

print(
    "Model saved as:",
    MODEL_PATH
)

# Final accuracy
training_accuracy = history.history["accuracy"][-1]
validation_accuracy = history.history["val_accuracy"][-1]

print(
    "Training Accuracy:",
    round(training_accuracy * 100, 2),
    "%"
)

print(
    "Validation Accuracy:",
    round(validation_accuracy * 100, 2),
    "%"
)

# Accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid()

plt.savefig(
    "accuracy_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: accuracy_graph.png"
)

# Loss graph
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid()

plt.savefig(
    "loss_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: loss_graph.png"
)

print("\n" + "=" * 50)
print("PROJECT TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)
