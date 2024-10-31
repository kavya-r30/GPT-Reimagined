import os
import torch

class Config:
    def __init__(self):
        self.n_layer = 2
        self.n_head = 2
        self.n_embd = 256
        self.embd_pdrop = 0.2
        self.resid_pdrop = 0.2
        self.attn_pdrop = 0.2
        self.dropout = 0.2
        self.compile = False
        self.num_workers = 0
        self.max_iters = 9435
        self.batch_size = 64
        self.block_size = 64
        self.vocab_size = 50257
        self.learning_rate = 2e-5
        self.betas = (0.9, 0.98)
        self.weight_decay = 3e-1
        self.grad_norm_clip = 1.0
        self.save_model_interval = self.max_iters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model_dir = os.path.join(current_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        self.log_dir = os.path.join(current_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.archive_log_dir = os.path.join(current_dir, 'archive_logs')
        os.makedirs(self.archive_log_dir, exist_ok=True)