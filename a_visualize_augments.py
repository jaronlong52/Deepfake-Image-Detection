import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

IMG_PATH = "test_images/from_test_set/fake.jpg"  # Path to the input image
NUM_AUGMENTATIONS = 5              # Number of augmented versions to generate
IMG_HEIGHT = 299                   # Height of the resized image
IMG_WIDTH = 299                    # Width of the resized image

# Define the data augmentation pipeline (same as in a_model.py)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

def visualize_augmentations(img_path, num_augmentations, img_size):
    """
    Visualizes the effect of the data augmentation pipeline on a single image.
    
    Args:
        img_path (str): Path to the input image.
        num_augmentations (int): Number of augmented versions to generate.
        img_size (tuple): Target size for the image (height, width).
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image {img_path} not found.")

    # Load the image
    original_img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(original_img)  # Values in range [0, 255]
    
    # Create a copy for display (normalize for visualization)
    display_img = img_array / 255.0  # Normalize to [0, 1] for display
    
    # Prepare the image for augmentation (keep in [0, 255] for the pipeline)
    img_batch = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Generate augmented images
    augmented_images = []
    for i in range(num_augmentations):
        aug_img = data_augmentation(img_batch, training=True)
        aug_img = tf.squeeze(aug_img, 0).numpy()  # Remove batch dimension
        
        # Debug: Print min and max pixel values to diagnose issues
        print(f"Augmented {i+1} - Min pixel value: {aug_img.min()}, Max pixel value: {aug_img.max()}")
        
        # Clip pixel values to [0, 255] to prevent invalid values
        aug_img = np.clip(aug_img, 0, 255)
        
        # Normalize to [0, 1] for display
        aug_img = aug_img / 255.0
        augmented_images.append(aug_img)

    # Create the figure and subplots
    fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(15, 5))
    
    # Add a main title to the entire figure
    fig.suptitle("Data Augmentation Visualization", fontsize=16, y=0.88)
    
    # Plot original image
    axes[0].imshow(display_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Plot augmented images
    for i, aug_img in enumerate(augmented_images):
        axes[i + 1].imshow(aug_img)
        axes[i + 1].set_title(f"Augmented {i + 1}")
        axes[i + 1].axis("off")

    # Adjust layout to prevent overlap with suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for the suptitle
    plt.show()

# Run the visualization with the specified configuration
if __name__ == "__main__":
    img_size = (IMG_HEIGHT, IMG_WIDTH)
    visualize_augmentations(IMG_PATH, NUM_AUGMENTATIONS, img_size)