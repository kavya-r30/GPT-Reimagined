import os
import json
import shutil
import requests
import regex as re
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pickle
import math
import time
import tiktoken
import torchinfo
from tqdm.auto import tqdm
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.base_weight, gain=self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(
                        self.grid_size + 1, self.in_features, self.out_features
                    )
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (
                    self.scale_spline
                    if not self.enable_standalone_scale_spline
                    else 1.0
                )
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.constant_(self.spline_scaler, self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape \
                (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid  # type: ignore
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given
        points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape \
                (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape \
                (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(
            splines, orig_coeff
        )  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0,
                batch - 1,
                self.grid_size + 1,
                dtype=torch.int64,
                device=x.device,
            )
        ]

        uniform_step = (
            x_sorted[-1] - x_sorted[0] + 2 * margin
        ) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = (
            self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        )
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(
                    self.spline_order, 0, -1, device=x.device
                ).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(
                    1, self.spline_order + 1, device=x.device
                ).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)  # type: ignore
        self.spline_weight.data.copy_(
            self.curve2coeff(x, unreduced_spline_output)
        )

    def regularization_loss(
        self, regularize_activation=1.0, regularize_entropy=1.0
    ):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as
        stated in the paper, since the original one requires computing
        absolutes and entropy from the expanded
        (batch, in_features, out_features) intermediate tensor, which is
        hidden behind the F.linear function if we want an memory
        efficient implementation.

        The L1 regularization is now computed as mean absolute value of the
        spline weights. The authors implementation also includes this term
        in addition to the sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        width,
        grid=3,
        k=3,
        noise_scale=0.1,
        noise_scale_base=1.0,
        scale_spline=1.0,
        base_fun=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        bias_trainable=True,
    ):
        super(KAN, self).__init__()
        self.grid_size = grid
        self.spline_order = k
        self.bias_trainable = bias_trainable 

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(width, width[1:]):
            self.layers.append(
                KANLinear(
                    in_features,out_features,grid_size=grid,spline_order=grid,
                    scale_noise=noise_scale,scale_base=noise_scale_base,scale_spline=scale_spline,
                    base_activation=base_fun,grid_eps=grid_eps,grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        B, C, T = x.shape

        x = x.view(-1, T)

        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)

        U = x.shape[1]

        x = x.view(B, C, U)

        return x

    def regularization_loss(
        self, regularize_activation=1.0, regularize_entropy=1.0
    ):
        return sum(
            layer.regularization_loss(
                regularize_activation, regularize_entropy
            )
            for layer in self.layers
        )
    
data_dir = os.path.join('data', 'tinyshakespeare')
input_file_path = os.path.join(data_dir, 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    os.makedirs(data_dir)
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for key, value in kwargs.items():
            setattr(self, key, value)

class CustomConfig(GPTConfig):
    n_layer = 2
    n_head = 2
    n_embd = 256
    embd_pdrop = 0.2
    resid_pdrop = 0.2
    attn_pdrop = 0.2
    dropout = 0.2
    compile = True
    device = device
    num_workers = 0
    max_iters = 9435
    batch_size = 64
    block_size = 64
    learning_rate = 3e-5
    betas = (0.9, 0.98)
    weight_decay = 3e-1
    grad_norm_clip = 1.0
    log_dir = '/kaggle/working/tensorboard_logs'

vocab_size = len(train_ids)
config = CustomConfig(vocab_size=enc.n_vocab)

data_dir = os.path.join('data', 'tinyshakespeare')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

class ShakespeareDataset(Dataset):
    def __init__(self, split, block_size=128, device_type='cuda'):
        assert split in {'train', 'test'}
        self.split = split
        self.block_size = block_size
        self.device_type = device_type
        self.data = train_data if split == 'train' else val_data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)) 

        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)
        else:
            x, y = x.to('cpu'), y.to('cpu')
        return x, y

# create dataset and dataloader
train_dataset = ShakespeareDataset('train', config.block_size, config.device)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=False)
test_dataset = ShakespeareDataset('test', config.block_size, config.device)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=False)

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = KAN([config.n_embd, 3 * config.n_embd])
        self.c_proj = KAN([config.n_embd, config.n_embd])
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.dropout = config.dropout
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("mask", 
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
            
    def forward(self, x):
        # batch_size, seq_len, emb_dim
        B, T, C = x.size() 

        # (batch_size, seq_len, emb_dim) --> (batch_size, seq_len, emb_dim * 3) --> (batch_size, seq_len, emb_dim)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (b, h, seq_len, d_k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (b, h, seq_len, d_k)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (b, h, seq_len, d_k)
        
        # (batch_size, n_heads, seq_len, single_head_dim) x (B, nh, single_head_dim, seq_len) -> (B, nh, seq_len, seq_len)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # (b, n_heads, seq_len, seq_len) @ (b, n_heads, seq_len, single_head_dim) --> (b, n_heads, seq_len, single_head_dim)
        y = att @ v 

        # (b, n_heads, seq_len, single_head_dim) --> (b, seq_len, n_heads, single_head_dim) --> (b, seq_len, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.ModuleDict(dict(
            c_fc = KAN([config.n_embd, 4 * config.n_embd]),
            c_proj = KAN([4 * config.n_embd, config.n_embd]),
            act = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        
        # (batch_size, seq_len, emb_dim)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = KAN([config.n_embd, config.vocab_size], bias_trainable=False)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, KAN, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # positional token, shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) 

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        # (b, t, n_embd) -- > # (b, t, vocab_size)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        # -1 at output will be ignored
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
    
class Trainer:
    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.device = config.device
        self.model = self.model.to(self.device)
        self.loss_batch = []

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        
        self.writer = SummaryWriter(log_dir=config.log_dir)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            # pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)
            self.loss_batch.append(self.loss.item())
            
            # Log loss to TensorBoard
            self.writer.add_scalar('Loss/train', self.loss.item(), self.iter_num)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow
            
            # Log iteration time to TensorBoard
            self.writer.add_scalar('Time/iter', self.iter_dt, self.iter_num)

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
                
        # Close the writer atfer the end of training
        self.writer.close()

def batch_end_callback(trainer):
    if trainer.iter_num % 250 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    
    if (trainer.iter_num+1) % 1000 == 0:
        avg_test_loss = evaluate_test_loss(trainer, model, config, test_loader)
        print('-'*64)
        print(f"Test Loss: {avg_test_loss: .5f}")
        print('-'*64)
    
    if (trainer.iter_num+1) % len(train_loader) == 0:
        epoch_num = (trainer.iter_num+1) / len(train_loader)
        avg_epoch_loss = sum(trainer.loss_batch) / len(train_loader)
        
        trainer.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch_num)
        
        avg_train_loss = evaluate_test_loss(trainer, model, config, train_loader)
        avg_test_loss = evaluate_test_loss(trainer, model, config, test_loader)
        
        print('-'*64)
        print(f"Loss for epoch {epoch_num} is: {avg_epoch_loss}")
        print(f"Train Loss: {avg_train_loss: .5f} | Test Loss: {avg_test_loss: .5f}")
        print('-'*64)
        trainer.loss_batch = []
        
def evaluate_test_loss(trainer, model, config, loader):  
    model.eval()
    total_test_loss = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for batch in tqdm(loader):
            # batch = [t.to(config.device) for t in batch]
            x, y = batch
            logits, loss = model(x, y)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / num_batches
#     print(f"Average Test Loss: {avg_test_loss:.5f}")

    trainer.writer.add_scalar('Loss/test', avg_test_loss, trainer.iter_num)

    model.train()  # Switch back to training mode
    return avg_test_loss
        
model = GPT(config).to(config.device)
# if config.compile:
#      model = torch.compile(model)
trainer = Trainer(config, model, train_dataset)

trainer.set_callback('on_batch_end', batch_end_callback)
trainer.run()

shutil.make_archive('/kaggle/working/tensorboard_logs', 'zip', config.log_dir)

def save_model(model, directory='/kaggle/working'):
  filename = (f"model.pth")
  filepath = os.path.join(directory, filename)

  torch.save(model.state_dict(), filepath)
  print(f"Model saved to {filepath}")

save_model(model)

def generate(model, idx, max_new_tokens, block_size=config.block_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, loss = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

text = 'Lord:\nRise! My people, conquer the north!'
sample_ids = torch.Tensor(enc.encode_ordinary(text)).long()
sample_ids = torch.unsqueeze(sample_ids, 0).to(config.device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(enc.decode(generate(model, sample_ids, max_new_tokens=2000)[0].tolist()))