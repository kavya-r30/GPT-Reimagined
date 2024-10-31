import os
import requests
import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data', 'tinyshakespeare')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# data_dir = os.path.join('data', 'tinyshakespeare')
input_file_path = os.path.join(data_dir, 'input.txt')

if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    # os.makedirs(data_dir, exist_ok=True)
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'test.bin'))

print(f"Train file path: {os.path.join(data_dir, 'train.bin')}")
print(f"Val file path: {os.path.join(data_dir, 'val.bin')}")

class ShakespeareDataset(Dataset):
    def __init__(self, split, block_size=64, device_type='cuda'):
        assert split in {'train', 'test'}
        self.split = split
        self.block_size = block_size
        self.device_type = device_type
        self.data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx: idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1: idx + 1 + self.block_size].astype(np.int64))

        if self.device_type == 'cuda':
            x, y = x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)
        else:
            x, y = x.to('cpu'), y.to('cpu')
        return x, y

def get_data_loaders(config):
    train_dataset = ShakespeareDataset('train', config.block_size, config.device)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=False)
    
    val_dataset = ShakespeareDataset('test', config.block_size, config.device)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=False)

    return train_loader, val_loader