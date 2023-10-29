import json


def information():
    with open('/data/tzeshinchen/research/gpt2/config.json', 'r') as f:
        config = json.load(f)
    print()
    print("*******************************************************")
    for i in config:
        print(i, " : ", config[i])
    print("*******************************************************")
    print()
