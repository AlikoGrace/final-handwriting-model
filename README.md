# Handwriting Dyslexia Detection

This project aims to detect the risk of dyslexia in children based on handwriting analysis. The model is trained to classify handwriting samples into three categories: normal, reversal, and corrected handwriting.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project uses a Convolutional Neural Network (CNN) model to classify handwriting images. The model is trained on a dataset containing images of normal, reversed, and corrected handwriting. The application provides an API to upload handwriting images and returns predictions indicating whether the handwriting is at risk of dyslexia.

## Dataset

The dataset consists of images collected from three sources:

- Uppercase Letters: NIST Special Database 19
- Lowercase Letters: Kaggle Dataset
- Additional Data: Dyslexic students at Seberang Jaya Primary School, Penang, Malaysia

The dataset contains:
- Normal class: 78,275 images
- Reversal class: 52,196 images
- Corrected class: 8,029 images

Each image is resized to 32x32 pixels.

## Model Architecture

The model is based on a modified LeNet-5 architecture with the following layers:
- Convolutional Layers
- Pooling Layers
- Batch Normalization
- Dropout
- Fully Connected Layers

The model was trained with various hyper-parameters to improve accuracy and reduce overfitting.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/handwriting-dyslexia-detection.git
    cd handwriting-dyslexia-detection
    ```

2. Set up a Python virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure you have TensorFlow installed. If not, you can install it:
    ```sh
    pip install tensorflow
    ```

5. Ensure you have Flask installed. If not, you can install it:
    ```sh
    pip install Flask
    ```

## Usage

### Running the Flask API

1. Ensure your trained model (`handwriting_classification_model.h5`) is in the project directory.

2. Run the Flask app:
    ```sh
    python app.py
    ```

The server will start on `http://127.0.0.1:5000`.

### Making Predictions

Use the `/upload` endpoint to upload images and get predictions.

### Example with `curl`

```sh
curl -X POST -F "file=@path_to_your_image.jpg" http://127.0.0.1:5000/upload
