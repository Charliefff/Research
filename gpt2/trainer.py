from input.embedding import encode, decode
from input.data_input import data_size
from model.GPT import BigramLanguageModel

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.nn import functional as F


class GPT2Trainer:
    def __init__(self, config):
        self.config = config
        # hyperparameters
        self.batch_size = self.config["batch_size"]
        self.block_size = self.config["block_size"]
        self.max_iters = self.config["max_iters"]
        self.eval_iters = self.config["eval_iters"]
        self.learning_rate = self.config["learning_rate"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = self.config["eval_iters"]
        self.n_embd = self.config["n_embd"]
        self.n_head = self.config["n_head"]
        self.n_layer = self.config["n_layer"]
        self.dropout = self.config["dropout"]
        self.vocab_size = self.config["vocab_size"]
        self.training_data = self.config["training_data"]
        self.training_size = self.config["training_size"]
        self.warnup_steps = self.config["warnup_steps"]

        self.log_dir = "logs/GPT2_" + \
            str(self.training_size) + '_' + str(self.n_layer)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        assert self.n_embd % self.n_head == 0

        self.training, self.validation = [], []

    def get_batch(self, split):
        data = self.training if split == 'train' else self.validation
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def load_data(self):
        with open(self.training_data, 'r', encoding='utf-8') as f:
            text = f.read()

        text = data_size(text, self.training_size)
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data))
        self.training = data[:n]
        self.validation = data[n:]

    def train(self):
        model = BigramLanguageModel(self.config)
        m = model.to(self.device)

        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate)
        initial_lr = self.learning_rate

        for iter in range(self.max_iters):
            if iter < self.warnup_steps:
                lr = initial_lr * (iter / self.warnup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if iter % self.eval_iters == 0:
                losses = self.estimate_loss(model)
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                self.writer.add_scalar(
                    'Loss/train_loss', losses['train'], iter)
                self.writer.add_scalar('Loss/val_loss', losses['val'], iter)

            xb, yb = self.get_batch('train')

            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iter % 25000 == 0 and iter != 0:
                FILE = '/data/tzeshinchen/research/gpt2/logs/checkpoints/'+str(self.training_size)+'_' + \
                    str(iter)+'_model_state_dict.pt'
                torch.save(model.state_dict(), FILE)
                print("save model")

        self.writer.close()
        FILE = '/data/tzeshinchen/research/gpt2/logs/checkpoints/'+str(self.training_size)+'_' + \
            str(self.n_layer)+'_model_state_dict.pt'
        torch.save(model.state_dict(), FILE)

    def information(self):

        print()
        print("*******************************************************")
        for i in self.config:
            print(i, " : ", self.config[i])
        print("*******************************************************")
        print()

    def main(self):
        self.information()
        self.load_data()
        self.train()
