import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# Define label mapping to match training setup
CLASS_MAPPING = {
    'real': {'label': 0, 'display': 'Real'},
    'fake': {'label': 1, 'display': 'Fake'}
}
CLASS_NAMES = ['real', 'fake']  # Matches training: real=0, fake=1
DISPLAY_LABELS = [CLASS_MAPPING[name]['display'] for name in CLASS_NAMES]

# Confirm label mapping
print("Label mapping:")
print(f"Class 0: {CLASS_NAMES[0]} → {DISPLAY_LABELS[0]}")
print(f"Class 1: {CLASS_NAMES[1]} → {DISPLAY_LABELS[1]}")
print(f"Prediction logic: > threshold → {DISPLAY_LABELS[1]}, < threshold → {DISPLAY_LABELS[0]}")

# Load model and threshold
model_path = "saved_model/deepfake_detector_corrected.keras"  # Updated to corrected model
threshold_path = "saved_model/optimal_threshold_corrected.npy"  # Updated to corrected threshold
model = load_model(model_path)
best_thresh = float(np.load(threshold_path)) if os.path.exists(threshold_path) else 0.5
print(f"Loaded model from: {model_path}")
print(f"Using threshold: {best_thresh:.2f}")

# Constants
IMG_HEIGHT, IMG_WIDTH = 299, 299

def predict_image_visual(img_path, model, img_size=(IMG_HEIGHT, IMG_WIDTH), true_label=None, threshold=0.5):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image {img_path} not found.")

    # Original image for display
    original_img = image.load_img(img_path)
    display_img = np.array(original_img) / 255.0

    # Preprocess image for model
    resized_img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(resized_img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

    # Prediction
    prediction = model.predict(img_preprocessed, verbose=0)[0][0]
    # Assumes training setup: real=0, fake=1, so > threshold means Fake (class 1)
    predicted_label = DISPLAY_LABELS[1] if prediction >= threshold else DISPLAY_LABELS[0]
    confidence = prediction if prediction >= threshold else 1 - prediction

    # Label formatting
    color = "red" if predicted_label == DISPLAY_LABELS[1] else "green"
    if true_label:
        title_text = f"Predicted: {predicted_label} ({confidence:.2f})\nActual: {true_label}"
    else:
        title_text = f"Predicted: {predicted_label} ({confidence:.2f})"

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Image
    ax[0].imshow(display_img)
    ax[0].axis("off")
    ax[0].set_title(title_text, fontsize=14, color=color)

    # Right: Confidence Bar Plot
    ax[1].bar(
        DISPLAY_LABELS,
        [1 - prediction, prediction],      # Real = 1–p(fake), Fake = p(fake)
        color=["green", "red"]              # (optional) Real in green, Fake in red
    )
    ax[1].axhline(threshold, color='black', linestyle='--', linewidth=1)
    ax[1].text(0.5, threshold + 0.02, f"Threshold ({threshold:.2f})", ha='center', fontsize=10, color="black")
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("Confidence")
    ax[1].set_title("Prediction Confidence", fontsize=12)
    ax[1].tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    plt.show()

# Example usage
predict_image_visual("test_images/from_test_set/fake.jpg", model, true_label="Fake", threshold=best_thresh)
# predict_image_visual("test_images/from_test_set/real.jpg", model, true_label="Real", threshold=best_thresh)