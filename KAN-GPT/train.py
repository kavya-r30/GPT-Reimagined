import os
import shutil

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
from tqdm.auto import tqdm
from collections import defaultdict

class Trainer:
    def __init__(self, config, model, train_loader, test_loader):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_loader = train_loader
        self.test_loader = test_loader
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

    def create_log_archive(self):
        archive_name = os.path.join(self.config.archive_log_dir, 'tensorboard_logs')
        shutil.make_archive(archive_name, 'zip', self.config.archive_log_dir)
        print(f"Logs archived to {archive_name}.zip")

    def save_model(self):
        filepath = os.path.join(self.config.model_dir, "model.pth")
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def run(self):
        model, config = self.model, self.config
        
        self.optimizer = model.configure_optimizers(config)

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(self.train_loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            logits, self.loss = model(x, y)
            self.loss_batch.append(self.loss.item())
            
            # Log loss to TensorBoard
            self.writer.add_scalar('Loss/train', self.loss.item(), self.iter_num)

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

            # if self.iter_num % config.save_model_interval == 0:
            #     self.save_model()

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
                
        self.save_model()
        
        # Close the writer atfer the end of training
        self.writer.close()
        self.create_log_archive()

def batch_end_callback(trainer):
    if trainer.iter_num % 250 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    
    if (trainer.iter_num+1) % 1000 == 0:
        avg_test_loss = evaluate_test_loss(trainer, trainer.model, trainer.config, trainer.test_loader)

        print('-'*64)
        print(f"Test Loss: {avg_test_loss: .5f}")
        print('-'*64)
    
    if (trainer.iter_num+1) % len(trainer.train_loader) == 0:
        epoch_num = (trainer.iter_num+1) / len(trainer.train_loader)
        avg_epoch_loss = sum(trainer.loss_batch) / len(trainer.train_loader)
        
        trainer.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch_num)
        
        avg_train_loss = evaluate_test_loss(trainer, trainer.model, trainer.config, trainer.train_loader)
        avg_test_loss = evaluate_test_loss(trainer, trainer.model, trainer.config, trainer.test_loader)
        
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
            batch = [t.to(config.device) for t in batch]
            x, y = batch
            logits, loss = model(x, y)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / num_batches
    # print(f"Average Test Loss: {avg_test_loss:.5f}")

    trainer.writer.add_scalar('Loss/test', avg_test_loss, trainer.iter_num)

    model.train()
    return avg_test_loss