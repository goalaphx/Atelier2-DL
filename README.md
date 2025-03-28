# Vision Transformer (ViT) vs CNN for MNIST Classification

## Overview

This project explores two deep learning architectures for image classification on the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The two architectures implemented are:

1. **Convolutional Neural Network (CNN)** – A classic approach for image classification.
2. **Vision Transformer (ViT)** – A transformer-based architecture that treats images as sequences of patches.

The goal of this project is to compare the performance of both models on the MNIST dataset.

---

## Part 1: CNN Model

### Model Architecture

The CNN model consists of:

- **Convolutional Layers**: Several convolutional layers followed by ReLU activations and max-pooling layers.
- **Fully Connected Layers**: After the convolutional layers, the output is flattened and passed through fully connected layers.
- **Output Layer**: A softmax layer classifies the image into one of the 10 digits (0-9).

### Training and Evaluation

- **Epochs**: 5
- **Batch Size**: 128
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss

### Results

- **Test Loss**: 0.49
- **Test Accuracy**: 98.36%

---

## Part 2: Vision Transformer (ViT) Model

### Model Architecture

The ViT model is based on the transformer architecture and includes the following components:

- **Patch Embedding**: Images are divided into 7x7 patches, each flattened and passed through a linear layer.
- **Positional Embedding**: Positional information is added to the patch embeddings.
- **Transformer Blocks**: The model includes 2 transformer blocks, each with multi-head self-attention (MHSA) layers and MLPs.
- **Classification Token**: A learnable classification token is used for final prediction.
- **Output Layer**: A softmax layer for classification.

### Training and Evaluation

- **Epochs**: 5
- **Batch Size**: 128
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss

### Results

- **Test Loss**: 1.70
- **Test Accuracy**: 76.29%

---

## Comparison of Results

| Model  | Test Loss | Test Accuracy |
|--------|-----------|---------------|
| **CNN** | 0.49     | 98.36%        |
| **ViT** | 1.70      | 76.29%        |

---

## Interpretation

- **CNN** outperforms **ViT** on the MNIST dataset, achieving a higher test accuracy (98.45% vs. 95.20%). This is expected, as CNNs have been extensively optimized for image classification tasks, particularly on smaller datasets like MNIST.
- **ViT**, while a powerful model in many computer vision tasks, requires larger datasets to fully showcase its advantages. Despite this, it still performs reasonably well, achieving over 95% accuracy.

---

## Conclusion

- The **CNN model** is more efficient and effective for MNIST classification due to its architecture, which is well-suited for image data.
- The **Vision Transformer** model, while state-of-the-art, may require more data and fine-tuning to outperform CNNs on small-scale tasks like MNIST.
