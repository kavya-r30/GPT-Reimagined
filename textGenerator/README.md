# Marvel GPT Language Model

This repository contains a GPT language model trained on Marvel movie scripts up until "Avengers: Endgame." The model is capable of generating Marvel-style dialogues and texts based on user input, utilizing a transformer-based architecture.

## Model Architecture

The model uses a standard transformer-based architecture with the following components:

- **Token Embedding**: Converts input tokens (characters) into embeddings of size `n_embd`.
- **Position Embedding**: Adds positional information to token embeddings for sequence processing.
- **Multi-Head Attention**: Six attention heads, each of size `n_embd // n_head`, for capturing dependencies across input sequences.
- **FeedForward Neural Network**: Processes each token independently after the attention layers.
- **Layer Normalization**: Applied after attention and feedforward blocks.
- **Language Model Head**: Outputs a prediction over the vocabulary at each position.

### Hyperparameters

- **Block Size (T)**: 64 tokens
- **Batch Size (B)**: 256
- **Embedding Dimension (n_embd)**: 384
- **Attention Heads (n_head)**: 6
- **Layers (n_layer)**: 6
- **Dropout**: 0.2
- **Learning Rate**: 3e-4

## Training

The model is trained on Marvel movie scripts, with characters encoded into integers and split into training (90%) and validation (10%) sets. The training process uses AdamW as the optimizer and tracks loss during evaluation intervals to monitor both training and validation performance.

### Loss Function

The model uses Cross-Entropy Loss to compute the difference between predicted and actual next tokens during training.

### Generation

Once trained, the model can generate text sequences by predicting the next token iteratively until the desired length is reached. The `generate` function allows specifying the number of new tokens to generate after a given input context.