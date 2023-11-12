from input.embedding import encode
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, config):
        super(BigramLanguageModel, self).__init__()
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.token_embedding_table = nn.Embedding(
            config["vocab_size"], config["n_embd"])
        self.position_embedding_table = nn.Embedding(
            config["block_size"], config["n_embd"])

        self.blocks = nn.Sequential(
            *[self.Block(config) for _ in range(config["n_layer"])])

        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    class Block(nn.Module):
        def __init__(self, config):
            super(BigramLanguageModel.Block, self).__init__()
            n_embd = config["n_embd"]
            n_head = config["n_head"]
            dropout = config["dropout"]
            block_size = config["block_size"]

            head_size = n_embd // n_head
            self.sa_head = BigramLanguageModel.MultiHeadAttention(
                n_embd, n_head, head_size, dropout, block_size)
            self.ffwd = BigramLanguageModel.FeedForward(n_embd, dropout)
            self.norm1 = nn.LayerNorm(n_embd)
            self.norm2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = self.sa_head(self.norm1(x)) + x
            x = self.ffwd(self.norm2(x)) + x
            return x

    class MultiHeadAttention(nn.Module):
        def __init__(self, n_embd, num_head, head_size, dropout, block_size):
            super(BigramLanguageModel.MultiHeadAttention, self).__init__()
            self.heads = nn.ModuleList([BigramLanguageModel.Head(
                n_embd, head_size, dropout, block_size) for _ in range(num_head)])
            self.proj = nn.Linear(head_size * num_head, n_embd)
            self.dropout = nn.Dropout(dropout)
            self.block_size = block_size

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

    class Head(nn.Module):
        def __init__(self, n_embd, head_size, dropout, block_size):
            super(BigramLanguageModel.Head, self).__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(
                torch.ones(block_size, block_size)))
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out

    class FeedForward(nn.Module):
        def __init__(self, n_embd, dropout):
            super(BigramLanguageModel.FeedForward, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, n_embd * 4),
                nn.ReLU(),
                nn.Linear(n_embd * 4, n_embd),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            return self.net(x)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next == encode['>']:
                idx = torch.cat((idx, idx_next), dim=1)
                return idx
            else:
                idx = torch.cat((idx, idx_next), dim=1)
        return idx
