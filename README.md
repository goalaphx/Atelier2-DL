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

- **CNN** outperforms **ViT** on the MNIST dataset, achieving a higher test accuracy (98.36% vs. 76.29%). This is expected, as CNNs have been extensively optimized for image classification tasks, particularly on smaller datasets like MNIST.
- **ViT**, while a powerful model in many computer vision tasks, requires larger datasets to fully showcase its advantages. Despite this, it still performs reasonably well, achieving over 75% accuracy.

---

## CNN: Faster and Less Resource-Intensive

The **Convolutional Neural Network (CNN)** has several advantages when it comes to efficiency and resource consumption:

- **Faster Training**: CNNs are optimized for tasks like image classification, especially for smaller datasets like MNIST. The convolutional layers are computationally less expensive than the self-attention mechanisms in transformers, which makes CNNs significantly faster to train.
- **Lower Memory Usage**: The number of parameters in a CNN is typically much smaller than in a Vision Transformer, especially when the transformer has multiple attention heads and large hidden dimensions. This results in lower memory consumption during both training and inference.
- **Hardware Efficiency**: CNNs require less GPU power and memory bandwidth. This makes them a better choice for resource-constrained environments, where computational resources (such as memory, GPU cores, and processing power) are limited.

### Hardware Resource Consumption

- **GPU Memory**: The CNN model uses significantly less GPU memory than the ViT model. While the ViT model requires substantial memory for storing patch embeddings, positional embeddings, and attention matrices, CNNs only require memory for convolutional weights, activations, and a few fully connected layers.
- **Training Speed**: On a typical GPU (e.g., NVIDIA Tesla T4 or similar), the CNN model typically trains much faster due to its smaller memory footprint and fewer parameters to process.
- **Inference**: The CNN model is also faster during inference, meaning predictions are made more quickly compared to ViT, which needs to process all patch embeddings and perform attention computations across the entire image.

---

## Conclusion

- The **CNN model** is more efficient and effective for MNIST classification due to its architecture, which is well-suited for image data.
- The **Vision Transformer** model, while state-of-the-art, may require more data and fine-tuning to outperform CNNs on small-scale tasks like MNIST. Additionally, ViT's computational requirements make it less suitable for smaller datasets, particularly when hardware resources are limited.
- **CNNs** are the ideal choice when hardware resources are limited, and faster training/inference is crucial, while **ViT** shines on larger, more complex datasets where computational resources can be leveraged effectively.

---
