import os


def process_file(args):
    filename, path = args
    repeat = 0
    unique = 0

    with open(os.path.join(path, filename), 'r') as f:
        smiles_lines = f.read().splitlines()

    for smiles in smiles_lines:
        if smiles in smiles_lines:
            repeat += 1
        else:
            unique += 1

    return filename, repeat, unique


def main():
    path = "/data/tzeshinchen/research/gpt2/output/Out/"

    Files = []
    total_repeat = []
    total_unique = []

    dir_list = os.listdir(path)
    total_files = len(dir_list)

    args_for_pool = [(filename, path) for filename in dir_list]

    for args in args_for_pool:

        filename, repeat, unique = process_file(args)

        Files.append(filename)
        total_repeat.append(repeat)
        total_unique.append(unique)

    with open("/data/tzeshinchen/research/evaluate/num/unique.txt", "w") as f:
        for i in range(total_files):
            f.write("Files: " + str(Files[i]) + "\n")
            f.write("Total repeated SMILES: " + str(total_repeat[i]) + "\n")
            f.write("Total unique SMILES: " + str(total_unique[i]) + "\n")
            f.write("Overall unique rate: " +
                    str(total_unique[i] / 50000 if total_files else 0) + "\n")
            f.write("\n")


if __name__ == "__main__":
    main()
