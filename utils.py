import os
import torch
import torch.nn as nn
from torch.nn import functional as F

def save_model(model, directory='models'):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, "model.pth")
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

