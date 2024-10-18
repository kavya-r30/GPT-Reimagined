# Indian Names Character-Level Language Model

This project implements a character-level language model for generating Indian names using PyTorch. The model predicts the next character in a sequence based on previous characters and is capable of generating entirely new names after training.

## Model Overview

The model consists of several key components:
- **Embedding Layer**: Maps each character (26 letters of the alphabet and `.`) to a 16-dimensional vector.
- **Linear Layers**: Fully connected layers that transform the input data.
- **BatchNorm1D Layers**: Used to stabilize and accelerate training.
- **Tanh Activation**: A non-linear activation function to introduce non-linearity into the model.
- **Sequential Model**: The model is structured as a `Sequential` class, allowing for easy stacking of layers.

### Model Architecture

The model follows this structure:
- Embedding: Maps characters to 16-dimensional vectors.
- Flatten: Converts multi-dimensional tensors into 2D tensors.
- 5 repeated blocks of: 
  - Linear -> BatchNorm -> Tanh
- Final output layer: Linear -> BatchNorm.

## Training Process

The model is trained on a dataset of Indian names. Each name is represented as a sequence of characters, and the task is to predict the next character given the previous characters.

### Loss Function
- **Cross Entropy Loss**: Used to measure the difference between the predicted character distribution and the actual character.

### Training Parameters
- **Learning Rate**: Starts at 0.1 for the first 10,000 iterations and is reduced to 0.01 for the remaining 10,000 iterations.
- **Batch Size**: 32
- **Number of Iterations**: 20,000

## Dataset

The dataset consists of Indian names provided in a CSV file (`Indian_Names.csv`). The dataset is split as follows:
- 80% for training
- 10% for validation
- 10% for testing

## Model Evaluation

The model's performance is evaluated on both the validation and test sets. The `split_loss()` function computes the loss for the training, validation, and test datasets.

### Loss Evaluation

During training, the model’s loss is monitored to ensure convergence and generalization.