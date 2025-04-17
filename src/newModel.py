# ------------------------- #
# 0. Import Required Modules
# ------------------------- #

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception, preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------- #
# 1. Dataset Setup
# ------------------------- #

BASE_DIR = "/path/to/deepfake-and-real-images"  # Replace with your dataset path
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = 32

# Load datasets from separate train, val, and test directories
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(BASE_DIR, "train"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(BASE_DIR, "val"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(BASE_DIR, "test"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Preprocess input with Xception preprocessing
def preprocess(images, labels):
    return preprocess_input(images), labels

train_ds = train_ds.map(preprocess).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)

# ------------------------- #
# 2. Build Model with XceptionNet
# ------------------------- #

base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze base

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------- #
# 3. Class Weight Calculation
# ------------------------- #

class_counts = {0: 0, 1: 0}
for _, labels in train_ds.unbatch():
    class_counts[int(labels.numpy())] += 1

total = sum(class_counts.values())
class_weight = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}
print("Class weights:", class_weight)

# ------------------------- #
# 4. Train Model (Initial Phase)
# ------------------------- #

EPOCHS_INITIAL = 5

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("deepfake_xception_best.h5", save_best_only=True)
]

history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_INITIAL,
    class_weight=class_weight,
    callbacks=callbacks
)

# ------------------------- #
# 5. Fine-Tune Top Layers
# ------------------------- #

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

EPOCHS_FINE = 5
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    class_weight=class_weight,
    callbacks=callbacks
)

model.save("deepfake_xception_finetuned.h5")

# ------------------------- #
# 6. Evaluate on Test Set
# ------------------------- #

test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Final Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# ------------------------- #
# 7. Plot Training History
# ------------------------- #

def plot_history(histories):
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for h in histories:
        acc += h.history['accuracy']
        val_acc += h.history['val_accuracy']
        loss += h.history['loss']
        val_loss += h.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

plot_history([history_initial, history_fine])

# ------------------------- #
# 8. Predict on New Image
# ------------------------- #

def predict_image(img_path, model, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    label = "Fake" if prediction[0] > 0.5 else "Real"
    print(f"The image is predicted to be: {label} (score: {prediction[0][0]:.2f})")

# Example usage:
# predict_image("/path/to/test_image.jpg", model)
