# Smart Waste Classification Using a Deep Neural Network

## 1. Problem Definition
Proper waste segregation supports recycling, reduces landfill waste, and protects the environment. This project classifies waste images into four categories: Paper, Plastic, Metal, and Organic.

## 2. Dataset
A shared dataset containing 10 images per category (40 images total) was used for the prototype. Images were organized into category folders and split into 70% training, 15% validation, and 15% testing sets. The assignment originally asks for 20–30 self-captured images per category; the current dataset does not meet that collection requirement.

## 3. Preprocessing and DNN Design
Images are resized to 128 × 128 pixels and normalized to the range 0–1. The DNN uses an input shape of 128 × 128 × 3, a Flatten layer, two Dense hidden layers with 128 and 64 neurons using ReLU activation, and a four-neuron Softmax output layer.

## 4. Training and Evaluation
The model is implemented using TensorFlow/Keras with the Adam optimizer and sparse categorical cross-entropy loss. Training and validation accuracy and loss are plotted. The test set is used for final evaluation.

## 5. Results
In the recorded run, test accuracy was 37.5% and test loss was 1.5663. The training accuracy became much higher than validation accuracy, suggesting possible overfitting. Since the test set contains only eight images, the result is highly uncertain and should not be treated as reliable real-world performance.

## 6. Challenges
**Challenge 1:** The dataset was small, limiting the examples available for learning. The limitation was documented, and the results were interpreted cautiously.

**Challenge 2:** Waste materials can look similar under different backgrounds and lighting. More varied images and data augmentation are potential improvements.

## 7. Conclusion
The project demonstrates a complete basic DNN image-classification workflow. Future work should use a larger and more diverse dataset and compare the DNN with a CNN.
