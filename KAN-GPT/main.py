import torch
import tiktoken

from model import GPT
from config import Config
from train import Trainer, batch_end_callback
from dataset_shakespeare import get_data_loaders
from generate import generate

def main():
    config = Config()
    model = GPT(config)

    train_loader, test_loader = get_data_loaders(config)
    config.max_iters = len(train_loader)

    trainer = Trainer(config, model, train_loader, test_loader)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    text = 'Lord:\nRise! My people, conquer the north!'
    enc = tiktoken.get_encoding("gpt2")
    sample_ids = torch.Tensor(enc.encode_ordinary(text)).long()
    sample_ids = torch.unsqueeze(sample_ids, 0).to(config.device)

    generated_indices = generate(model, sample_ids, max_new_tokens=2000, block_size=config.block_size)
    print(enc.decode(generated_indices[0].tolist()))

if __name__ == '__main__':
    main()