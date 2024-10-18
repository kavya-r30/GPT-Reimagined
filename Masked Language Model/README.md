# Masked Language Model (MLM) Using PyTorch

This project implements a Masked Language Model (MLM) using PyTorch. The model is designed to learn from a dataset of different novels, utilizing the transformer architecture for text generation tasks. The model incorporates multi-head attention mechanisms and employs a token masking strategy to enhance training.

## Requirements

- PyTorch
- NumPy
- Matplotlib
- `tiktoken` for GPT-2 tokenization

To install `tiktoken`, run:
pip install tiktoken

# Overview
The model architecture consists of several components:

- **Tokenization**: The `tiktoken` library is used to encode and decode text data.
- **Multi-Head Attention**: The model features a multi-head attention mechanism to capture dependencies in the input data.
- **FeedForward Neural Network**: Each attention block is followed by a feedforward network for further processing.
- **Masking**: A masking mechanism is implemented to randomly mask tokens during training, allowing the model to learn to predict masked tokens based on their context.

# Model Components

## Tokenization
The `tiktoken` library handles text encoding and decoding. The encoding function can be defined as follows:

## Neural Network Architecture
The following classes define the architecture of the MLM:

- **Head**: Implements a single attention head.
- **MultiHeadAttention**: Combines multiple attention heads.
- **FeedForward**: Defines a feedforward network with ReLU activations.
- **Block**: A composite block that contains multi-head attention and feedforward layers.
- **Encoder**: The main model class that integrates embeddings, transformer blocks, and output logits.

## Model Training
The model is trained using the following parameters:

**Vocabulary Size**: 50258
**Embedding Dimension**: 256
**Batch Size**: 256
**Block Size**: 64
**Learning Rate**: 1e-3
**Number of Heads**: 8
**Number of Blocks**: 4
**Maximum Iterations**: 5000
**Mask Probability**: 0.15
Training involves iterating through batches of data, applying token masking, and optimizing the model parameters using the AdamW optimizer.

## Usage
To run the training process, ensure you have a text file containing the novels, and adjust the file path accordingly:

`with open('/path/to/your/textdata.txt', 'r', encoding='utf-8') as f:
    text = f.read()`

`data = torch.tensor(encode(text), dtype=torch.long, device=device)`

Train the model using the specified parameters and save the model state after training:

`torch.save(model.state_dict(), 'model.pth')`