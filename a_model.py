import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception, preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import seaborn as sns

# ------------------------- #
# 0. Quick Test Toggle
# ------------------------- #
QUICK_TEST = False

# ------------------------- #
# 1. Configurable Parameters
# ------------------------- #
BASE_DIR     = "data"
IMG_HEIGHT   = 299
IMG_WIDTH    = 299
BATCH_SIZE   = 32 if not QUICK_TEST else 16
SEED         = 123

EPOCHS_INITIAL = 15 if not QUICK_TEST else 2
EPOCHS_FINE    = 30 if not QUICK_TEST else 4 #model will quit earlier than 30 if no improvement

# Dataset size limits for quick testing
LIMIT_TRAIN = 300
LIMIT_VAL   = 75
LIMIT_TEST  = 75

# ------------------------- #
# 2. Dataset Setup
# ------------------------- #
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(BASE_DIR, "train"),
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=['real', 'fake']  # Match subdirectory names (lowercase), real=0, fake=1
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(BASE_DIR, "train"),
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=['real', 'fake']
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(BASE_DIR, "test"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=['real', 'fake']
)

# Store class names before applying transformations
CLASS_NAMES = train_ds.class_names
print("Class names:", CLASS_NAMES)

# Apply limits if quick test is enabled
if QUICK_TEST:
    train_ds = train_ds.take(LIMIT_TRAIN // BATCH_SIZE)
    val_ds   = val_ds.take(LIMIT_VAL   // BATCH_SIZE)
    test_ds  = test_ds.take(LIMIT_TEST // BATCH_SIZE)

# Data augmentation and preprocessing pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

def prep(ds, augment=False):
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    return ds.map(lambda x, y: (preprocess_input(x), y))

train_ds = prep(train_ds, augment=True)
val_ds   = prep(val_ds)
test_ds  = prep(test_ds)

train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# ------------------------- #
# 3. Build and Compile Model
# ------------------------- #
base_model = Xception(weights='imagenet', include_top=False,
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.01)
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)
model.summary()

# ------------------------- #
# 4. Compute Class Weights
# ------------------------- #
class_counts = {0: 0, 1: 0}
for _, labels in train_ds.unbatch():
    class_counts[int(labels.numpy())] += 1
total = sum(class_counts.values())
class_weight = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1]),
}
print(f"Class weights: {class_weight}")
print(f"Class counts: {class_counts}")

# ------------------------- #
# 5. Initial Training Phase
# ------------------------- #
callbacks_initial = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7),
]

print(f"🔰 Initial training for {EPOCHS_INITIAL} epochs")
history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_INITIAL,
    class_weight=class_weight,
    callbacks=callbacks_initial
)

# ------------------------- #
# 6. Fine-Tuning Phase
# ------------------------- #
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-5),
    loss=loss_fn,
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print(f"🔧 Fine‑tuning for {EPOCHS_FINE} epochs")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    class_weight=class_weight,
    callbacks=callbacks_initial
)

# ------------------------- #
# 7. Evaluation and Plots
# ------------------------- #
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_ds)
print(f"✅ Test  acc: {test_acc:.3f}, prec: {test_prec:.3f}, recall: {test_rec:.3f}")

# Confusion matrix
def plot_confusion_matrix(dataset, model):
    y_true, y_pred = [], []
    for images, labels in dataset:
        predictions = model.predict(images) > 0.5
        y_true.extend(labels.numpy())
        y_pred.extend(predictions.astype(int).flatten())
    cm = confusion_matrix(y_true, y_pred)
    # Capitalize labels for readability
    display_labels = [name.capitalize() for name in CLASS_NAMES]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion_matrix(test_ds, model)

# Plot training history
def plot_training_history(history_list):
    """
    Plots accuracy and loss across epochs using multiple Keras history objects.
    Useful for visualizing both initial training and fine-tuning phases.
    """
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for history in history_list:
        acc += history.history.get('accuracy', [])
        val_acc += history.history.get('val_accuracy', [])
        loss += history.history.get('loss', [])
        val_loss += history.history.get('val_loss', [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_training_history([history_initial, history_fine])

# ------------------------- #
# 8. Single Image Prediction
# ------------------------- #
def predict_image_visual(img_path, model, img_size=(IMG_HEIGHT, IMG_WIDTH), true_label=None):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image {img_path} not found.")
    
    # Load original image for display
    original_img = image.load_img(img_path)  # Don't resize here
    display_img = np.array(original_img) / 255.0  # Normalize for display

    # Preprocess resized image for model prediction
    resized_img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(resized_img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)

    # Predict
    prediction = model.predict(img_preprocessed)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    predicted_label = CLASS_NAMES[1] if prediction > 0.5 else CLASS_NAMES[0]

    # Capitalize for display
    predicted_label_display = predicted_label.capitalize()
    display_labels = [name.capitalize() for name in CLASS_NAMES]

    # Optional: set true label for visual comparison
    if true_label is not None:
        label_text = f"Predicted: {predicted_label_display} ({confidence:.2f})\n Actual: {true_label}"
    else:
        label_text = f"Predicted: {predicted_label_display} ({confidence:.2f})"

    # Plot image and confidence
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(display_img)
    ax[0].set_title(label_text, fontsize=12)
    ax[0].axis("off")

    ax[1].bar(display_labels, [1 - prediction, prediction], color=["green", "red"])
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel("Confidence")
    ax[1].set_title("Prediction Confidence")

    plt.tight_layout()
    plt.show()

predict_image_visual("test_images/from_test_set/fake.jpg", model, img_size=(299, 299), true_label="Fake")
predict_image_visual("test_images/from_test_set/real.jpg", model, img_size=(299, 299), true_label="Real")
predict_image_visual("test_images/from_test_set/fake2.jpg", model, img_size=(299, 299), true_label="Fake")
predict_image_visual("test_images/from_test_set/real2.jpg", model, img_size=(299, 299), true_label="Real")
predict_image_visual("test_images/from_test_set/fake3.jpg", model, img_size=(299, 299), true_label="Fake")
predict_image_visual("test_images/from_test_set/real3.jpg", model, img_size=(299, 299), true_label="Real")

# ------------------------- #
# 9. Graphs
# ------------------------- #
def plot_roc_curve(dataset, model):
    y_true = []
    y_scores = []

    for images, labels in dataset:
        preds = model.predict(images).flatten()
        y_scores.extend(preds)
        y_true.extend(labels.numpy())

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Capitalize for display
    display_labels = [name.capitalize() for name in CLASS_NAMES]

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic ({display_labels[1]} vs {display_labels[0]})')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_roc_curve(test_ds, model)

def plot_precision_recall(dataset, model):
    y_true = []
    y_scores = []

    for images, labels in dataset:
        preds = model.predict(images).flatten()
        y_scores.extend(preds)
        y_true.extend(labels.numpy())

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # Capitalize for display
    display_labels = [name.capitalize() for name in CLASS_NAMES]

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'AP = {ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({display_labels[1]} vs {display_labels[0]})')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_precision_recall(test_ds, model)

def plot_prediction_distribution(dataset, model):
    preds_real, preds_fake = [], []
    total = 0

    for images, labels in dataset:
        total += len(labels)
        preds = model.predict(images).flatten()
        for p, label in zip(preds, labels.numpy()):
            if label == 0:
                preds_real.append(p)
            else:
                preds_fake.append(p)

    print(f"Total predictions processed: {total}")

    # Capitalize for display
    display_labels = [name.capitalize() for name in CLASS_NAMES]

    plt.hist(preds_real, bins=30, alpha=0.6, label=display_labels[0], color='green')
    plt.hist(preds_fake, bins=30, alpha=0.6, label=display_labels[1], color='red')
    plt.axvline(0.5, color='black', linestyle='--', label='Decision Boundary')
    plt.xlabel(f"Predicted Probability ({display_labels[1]})")
    plt.ylabel("Frequency")
    plt.title("Prediction Score Distribution")
    plt.legend()
    plt.show()

plot_prediction_distribution(test_ds, model)

# ------------------------- #
# 10. Metrics and Calibration
# ------------------------- #
y_true, y_scores = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs).flatten()
    y_scores.extend(preds)
    y_true.extend(labels.numpy())
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Raw metrics
test_pred = (y_scores > 0.5).astype(int)
print("Accuracy:", np.mean(test_pred == y_true))
print("Precision:", precision_recall_curve(y_true, y_scores))
roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
ap = average_precision_score(y_true, y_scores)
print(f"ROC AUC: {roc_auc:.3f}, AP: {ap:.3f}")

# Threshold optimization
prec, rec, thresh = precision_recall_curve(y_true, y_scores)
f1s = 2 * prec * rec / (prec + rec + 1e-8)
best = np.nanargmax(f1s[:-1])
best_thresh = thresh[best]
print(f"Best F1={f1s[best]:.3f} at threshold={best_thresh:.2f}")

# Calibration plot
prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Predicted probability')
plt.ylabel('True frequency')
plt.title('Reliability Diagram')
plt.show()

# ------------------------- #
# 11. Save Model and Threshold
# ------------------------- #
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/deepfake_detector_corrected.keras")
np.save("saved_model/optimal_threshold_corrected.npy", best_thresh)
print("✅ Corrected model and threshold saved successfully!")