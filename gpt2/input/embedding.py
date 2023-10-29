import json
with open("/data/tzeshinchen/research/gpt2/config.json", "r") as f:
    config = json.load(f)

encoding_dict = config["enocding_dict"]
decoding_dict = config["decoding_dict"]

with open(encoding_dict, 'r') as f:
    stoi = json.load(f)

with open(decoding_dict, 'r') as f:
    itos = json.load(f)


def encode(text):
    return [stoi[c] for c in text]


def decode(text):
    return ''.join([itos[str(i)] for i in text])
