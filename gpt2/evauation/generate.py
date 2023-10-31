from input.embedding import encode, decode
from input.data_input import data_size
from model.GPT import BigramLanguageModel
from information.print import information
import json

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
n_layer = 3
dropout = 0.2
path = "/data/tzeshinchen/research/gpt2/logs/weight/10000_3_model_state_dict.pt"
# ------------


torch.manual_seed(1337)
with open('/data/tzeshinchen/deep_learning/transformer/gpt2/data/smile_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

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
