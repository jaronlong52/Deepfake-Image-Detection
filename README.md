# Deepfake Image Detection

This project implements a deepfake detection model using a pre-trained Xception network, fine-tuned for classifying images as "Real" or "Fake." The model achieves high performance, with a test accuracy of 89.61%, precision of 93.26%, and recall of 85.56% (based on fine-tuning results).

### Slides Explaining the Design and Functionality

- https://docs.google.com/presentation/d/1HQdNtWehNQQF1kzP-72kJ_JLydQuFEioN4DNc0UVQn4/edit?slide=id.p#slide=id.p

# Setup and Usage Instructions

### Download and Install Python

- https://www.python.org/downloads/

### Clone the Repo

- https://github.com/jaronlong52/Deepfake-Image-Detection.git

### Create virtual environment

    python -m venv venv

### Activate virtual environment

    venv\Scripts\activate

### Install requirements

    pip install -r requirements.txt

- If you are missing requirements after this, look in all_packages.txt

### Link to Saved Model (trained)

- Download the files and place them in a folder titled "saved_model"
- https://drive.google.com/file/d/1PYXmxBpSiXyvGb6Piebi0OM2p5xMBpDa/view?usp=sharing

### Link to Dataset on Kaggle

- Download the dataset, unzip it, and place the dataset in the project directory.
- The model is set up to not use the entire dataset for the sake of time but it can be altered to include the entire set.
- The model is set up with the expectation that the data is stored in a directory called "data" with subdirectories "test" and "train, each of which have subdirectories "fake" and "real" (exact spelling).
  - The model splits data from the training set to use for validation, it can be altered to use the full dataset including the validation set.

```
data/
├── test/
│   ├── fake/
│   └── real/
└── train/
    ├── fake/
    └── real/
```

## Retraining the Model

- The model can be retrained as is using the CPU or alterations can be made to use your systems GPU (will require additional installations). I was unable to get this to work but it can speed up training time.

### Run the model

- Before running the model, set `QUICK_TEST = True` at the top of the file to test the model on your system. This takes far less time and allows for debugging any issues.

      python -m a_model

- Running the model will take some time and, towards the end of the process, will create graphs and other visuals about the data. These will need to be saved manually and the model will not continue until the image is closed.

## Testing the Model

- The tester file is a short script that utilizes the saved model to test images. Change the path in the method call at the bottom of the script to test different images.
- Some additional test images from the unused validation set are in a folder called test_images. The subdirectory "extra_new_images" contains images from the unused validation set from the Kaggle dataset.
- The visuals generated when testing images will need to be saved manually.

### Run the Test File

    python -m a_tester

## Other Tools

### Visualize Augmentations

- Alter the image path in the a_visualize_augments.py file and and run it to generate a visual to show how each image is augmented during training.

        python -m a_visualize_augments
