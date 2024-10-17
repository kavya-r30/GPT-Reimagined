import os
import torch
import torch.nn.functional as F
import tiktoken

def generate(model, idx, max_new_tokens, block_size=64):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

if __name__ == "__main__":
    from model import GPT
    from config import Config

    config = Config()
    model = GPT(config)

    model_filepath = os.path.join(config.model_dir, "model.pth")
    model.load_state_dict(torch.load(model_filepath))

    text = 'Lord:\nRise! My people, conquer the north!'
    enc = tiktoken.get_encoding("gpt2")
    sample_ids = torch.Tensor(enc.encode_ordinary(text)).long()
    sample_ids = torch.unsqueeze(sample_ids, 0).to(config.device)

    generated_indices = generate(model, sample_ids, max_new_tokens=10, block_size=config.block_size)
    print(enc.decode(generated_indices[0].tolist()))