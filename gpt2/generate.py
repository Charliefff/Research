import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F



# hyperparameters
batch_size = 16
block_size = 256
max_iters = 2000
eval_interval = 10
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 24
dropout = 0.2
path = "/data/tzeshinchen/deep_learning/transformer/gpt2/checkpoint/24_7000_model_state_dict.pt"
# ------------


torch.manual_seed(1337)
with open('/data/tzeshinchen/deep_learning/transformer/gpt2/data/smile_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# 找出多少字
chars = sorted(list(set(text)))
chars.append('<')
chars.append('>')
vocab_size = len(chars)
# 建立字典
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 字串到int
text = '\n'.join(['<' + line + '>' for line in text.strip().split('\n')])


def encode(s): return [stoi[c] for c in s]
#  int 到字串
def decode(l): return ''.join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


class Head(nn.Module):

    """ self attention layer """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.querry = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.querry(x)  # (B,T,C)
        v = self.value(x)  # (B,T,C)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v  # (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """ 多注意力機制"""

    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out  # (B,T,C)


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connection
        x = self.sa_head(self.norm1(x)) + x
        x = self.ffwd(self.norm2(x)) + x
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 根據位置編碼
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            if idx_next == stoi['>']:
                idx = torch.cat((idx, idx_next), dim=1)
                return idx
            else:
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
model.load_state_dict(torch.load(path))
print('Finish loading Model GPTLanguageModel')

# evaluation = 10000
# output = []
# for i in range(evaluation):
#     input_text = '<' + 'CN(C(=O)'
#     data = torch.tensor(encode(input_text),
#                         dtype=torch.long).unsqueeze(0).to(device)
#     generated_text = decode(m.generate(
#         data, max_new_tokens=2000)[0].tolist())
#     generated_text = generated_text.replace('<', '')
#     generated_text = generated_text.replace('>', '')
#     output.append(generated_text)

# with open('/data/tzeshinchen/deep_learning/transformer/gpt2/data/output/output_6_2000.txt', 'w', encoding='utf-8') as f:
#     for item in output:
#         f.write("%s\n" % item)

# 手動輸入
while True:
    try:
        # get user input
        input_text = input("Please enter the context text: ")
        input_text = '<' + input_text
        data = torch.tensor(encode(input_text),


                            dtype=torch.long).unsqueeze(0).to(device)
        generated_text = decode(m.generate(
            data, max_new_tokens=2000)[0].tolist())
        print(generated_text)

    except KeyboardInterrupt:
        print("\nExiting the program.")
        break
