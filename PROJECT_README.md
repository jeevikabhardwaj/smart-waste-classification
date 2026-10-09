# Smart Waste Classification Using a Deep Neural Network

## Project Overview
This project uses a Deep Neural Network (DNN) to classify waste images into four categories: Paper, Plastic, Metal, and Organic. It demonstrates how deep learning can support waste segregation on a college campus.

## Dataset
The shared dataset contains 40 images, with 10 images per category. The dataset is divided into training (70%), validation (15%), and testing (15%) sets.

Note: The assignment requests 20–30 self-captured images per category. This prototype uses a shared dataset and does not meet that specific collection requirement.

## Model Architecture
- Input: RGB images resized to 128 × 128 pixels.
- Rescaling: Pixel values are converted from 0–255 to 0–1.
- Flatten layer: Converts image pixels into a one-dimensional vector.
- Hidden layer 1: 128 neurons using ReLU activation.
- Hidden layer 2: 64 neurons using ReLU activation.
- Output layer: Four neurons using Softmax activation.

**Optimizer:** Adam  
**Loss function:** Sparse categorical cross-entropy  
**Framework:** TensorFlow/Keras

## Results
- Test accuracy: 37.5%
- Test loss: 1.5663

The model's training accuracy was higher than its validation accuracy, suggesting possible overfitting. Since the test set contains only eight images, the result is uncertain and should not be treated as reliable real-world performance.

## Technologies Used
Python, TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn, Google Colab, and GitHub.

## How to Run
1. Download or clone this repository.
2. Extract `WasteDataset.zip`.
3. Place the four category folders inside a directory named `dataset`.
4. Install the required libraries using `pip install -r requirements.txt`.
5. Run `python train.py`.

The script trains and evaluates the model and generates the result graphs.

## Challenges
1. **Limited data:** Only 10 images per category were available, limiting the model's ability to generalize.
2. **Visual variation:** Different backgrounds and lighting conditions can affect classification. More varied images and data augmentation could improve future results.

## Conclusion
This project demonstrates a basic DNN workflow for waste image classification. A larger dataset and further model improvements are needed for dependable real-world use.
