# Smart Waste Classification Using a Deep Neural Network

## Overview
This beginner-level project trains a Deep Neural Network (DNN) to classify waste images into four categories:

- Paper
- Plastic
- Metal
- Organic

The model uses TensorFlow/Keras and a simple fully connected neural network.

## Dataset
The project was tested with a shared dataset containing 10 images per category (40 images total). The images are split into training (70%), validation (15%), and testing (15%) sets by the training script.

**Important limitation:** this is a very small dataset. In the recorded run, test accuracy was **37.5%** (3 correct predictions out of 8 test images), with test loss **1.5663**. These results are not reliable estimates of real-world performance because the test set contains only eight images. Training accuracy was higher than validation/test performance, suggesting possible overfitting.

## Model architecture
1. Input image: 128 × 128 × 3 RGB
2. Rescaling: pixel values from 0–255 to 0–1
3. Flatten layer
4. Dense layer: 128 neurons, ReLU
5. Dense layer: 64 neurons, ReLU
6. Output layer: 4 neurons, Softmax

**Optimizer:** Adam  
**Loss function:** Sparse categorical cross-entropy  
**Metric:** Accuracy

## Repository structure
```text
smart-waste-classification/
├── dataset/                  # Add the four category folders here
│   ├── Paper/
│   ├── Plastic/
│   ├── Metal/
│   └── Organic/
├── results/                  # Training graphs and confusion matrix
├── train.py
├── requirements.txt
├── report.md
└── README.md
```

## How to run
1. Install Python 3.10 or 3.11.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the images inside `dataset/Paper`, `dataset/Plastic`, `dataset/Metal`, and `dataset/Organic`.
4. Run:
   ```bash
   python train.py
   ```

The script creates train/validation/test splits, trains the model, evaluates it, saves plots to `results/`, and saves the trained model as `waste_classifier.keras`.

## Limitations and future improvements
- Increase the number and variety of correctly labelled images.
- Use data augmentation carefully.
- Try a Convolutional Neural Network (CNN), which is more suitable for image classification.
- Evaluate on a larger, independent test set.

## Academic note
The dataset used for the current run was shared by another person. It should not be described as personally photographed data.
