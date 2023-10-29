from input.embedding import encode, decode
from input.data_input import data_size
from model.GPT import BigramLanguageModel
from information.print import information
import json

import tensorboard
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.nn import functional as F


with open("/data/tzeshinchen/research/gpt2/config.json", 'r') as f:
    config = json.load(f)

# hyperparameters
batch_size = config["batch_size"]
block_size = config["block_size"]
max_iters = config["max_iters"]
eval_iters = config["eval_iters"]
learning_rate = config["learning_rate"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = config["eval_iters"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
dropout = config["dropout"]
vocab_size = config["vocab_size"]
training_data = config["training_data"]
training_size = config["training_size"]
warnup_steps = config["warnup_steps"]
# ------------

log_dir = "logs/GPT2_" + str(training_size) + '_' + str(n_layer)
writer = SummaryWriter(log_dir=log_dir)

assert n_embd % n_head == 0
torch.manual_seed(1337)
information()

with open(training_data, 'r', encoding='utf-8') as f:
    text = f.read()

text = data_size(text, training_size)


# Train

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
# 平均各個batch的loss
def estimate_loss():
    out = {}
    # 模型轉成評估階段不會dropout or batchnorm
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        # 轉回訓練階段
    model.train()
    return out


print('start def model')
model = BigramLanguageModel()
m = model.to(device)
# model.load_state_dict("")

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
initial_lr = learning_rate

for iter in range(max_iters):
    if iter < warnup_steps:
        lr = initial_lr * (iter / warnup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        writer.add_scalar('Loss/train_loss', losses['train'], iter)
        writer.add_scalar('Loss/val_loss', losses['val'], iter)

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


writer.close()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
FILE = '/data/tzeshinchen/research/gpt2/logs/weigth/'+str(training_size)+'_' + \
    str(n_layer)+'_model_state_dict.pt'
torch.save(model.state_dict(), FILE)
