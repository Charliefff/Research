import json
import math
import numpy as np



encoding_dict = "/data/tzeshinchen/research/gpt2/tokenizer/encoder.json"
decoding_dict = "/data/tzeshinchen/research/gpt2/tokenizer/encoder.json"


with open(encoding_dict, 'r') as f:
    stoi = json.load(f)

with open(decoding_dict, 'r') as f:
    itos = json.load(f)


def encode(text, Segment=False, Position=False):

    Encoding, Segment_matrix, Position_matrix = [], [], []

    for num, token in enumerate(text):
        Encoding.append(stoi[token])
        Segment_matrix.append(math.ceil((num+1)/90))
        Position_matrix.append((num+1) % 90)

    if Segment:
        Encoding = np.sum([Encoding, Segment_matrix], axis=0)
        print("Finish segment encoding")

    if Position:
        Encoding = np.sum([Encoding, Position_matrix], axis=0)
        print("Finish Position encoding")

    return Encoding


def decode(text):
    return ''.join([itos[str(i)] for i in text])
