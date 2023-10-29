import json

with open('/data/tzeshinchen/research/dataset/smiles_pad.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# 建立字典
special = ['U']
stoi = {ch: i for i, ch in enumerate(chars+special)}
itos = {i: ch for i, ch in enumerate(chars+special)}

with open("./encoder.json", "w") as f:
    json.dump(stoi, f)

with open("./decoder.json", "w") as f:
    json.dump(itos, f)
