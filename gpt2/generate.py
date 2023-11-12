from input.embedding import encode, decode
from input.data_input import data_size
from model.GPT import BigramLanguageModel

import sys
from tqdm import trange

import torch

from torch.nn import functional as F


# hyperparameters
evaluation = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = "/data/tzeshinchen/research/gpt2/logs/weight/MolGPT_model_state_dict.pt"
out_dir = "/data/tzeshinchen/research/gpt2/output/MolGPT.txt"
# ------------

model = BigramLanguageModel()
m = model.to(device)
model.load_state_dict(torch.load(path))

type = sys.argv[1] if len(sys.argv) > 1 else "Auto"

print()
print('Finish loading Model GPTLanguageModel')
print("Type: ", type)
print("*******************************************************************************************")
print("Start generating compounds")
print()

if type == "Auto":
    output = []
    for i in trange(evaluation):
        input_text = '<'
        data = torch.tensor(encode(input_text),
                            dtype=torch.long).unsqueeze(0).to(device)
        generated_text = decode(m.generate(
            data, max_new_tokens=2000)[0].tolist())
        generated_text = generated_text.replace('<', '')
        generated_text = generated_text.replace('>', '')
        output.append(generated_text)

    with open(out_dir, 'w', encoding='utf-8') as f:
        for item in output:
            f.write("%s\n" % item)

elif type == "Manual":
    # 手動輸入
    while True:
        try:
            # get user input
            input_text = input("Please enter the context text: ")
            input_text = '<' + input_text
            data = torch.tensor(encode(input_text),
                                dtype=torch.long).unsqueeze(0).to(device)
            generated_text = decode(m.generate(

                data, max_new_tokens=100)[0].tolist())
            print(generated_text)

        except KeyboardInterrupt:
            print("\nExiting the program.")
            break

print()
print("Finish generating compounds")
print("*******************************************************************************************")
