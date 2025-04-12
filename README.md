# Deepfake Detection Using XceptionNet

This project uses **PyTorch** and **XceptionNet** to detect deepfake images. The dataset used is from Kaggle's Deepfake dataset, and the model is trained to classify images as either "real" or "fake".

## Setup

1.  Install the dependencies:

        pip install -r requirements.txt

2.  Download the data from the following link and place in the project directory

        https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images?resource=download-directory

## Training the Model

To train the model, run the following command:

        python src/train.py

## Testing the Model

To test the model, run:

        python src/test.py
