import os
from multiprocessing import Pool, cpu_count


def process_file(args):
    filename, path, Ans_path = args
    repeat = 0
    novety = 0

    with open(os.path.join(path, filename), 'r') as f:
        smiles_lines = f.read().splitlines()

    with open(Ans_path, 'r') as f:
        Ans_lines = set(f.read().splitlines())

    for num, smiles in enumerate(smiles_lines):
        if smiles in Ans_lines:
            repeat += 1
        else:
            novety += 1
        if (num + 1) % 1000 == 0:
            print("Processed", num + 1, "SMILES")

    return filename, repeat, novety


def main():
    path = "/data/tzeshinchen/research/gpt2/output/Out/"
    Ans_path = "/data/tzeshinchen/research/dataset/smile_input.txt"
    Files = []
    total_repeat = []
    total_novety = []

    dir_list = os.listdir(path)
    total_files = len(dir_list)

    args_for_pool = [(filename, path, Ans_path) for filename in dir_list]

    for args in args_for_pool:

        filename, repeat, novety = process_file(args)

        Files.append(filename)
        total_repeat.append(repeat)
        total_novety.append(novety)

    with open("/data/tzeshinchen/research/evaluate/num/validity.txt", "w") as f:
        for i in range(total_files):
            f.write("Files: " + str(Files[i]) + "\n")
            f.write("Total repeated SMILES: " + str(total_repeat[i]) + "\n")
            f.write("Total novel SMILES: " + str(total_novety[i]) + "\n")
            f.write("Overall novelty rate: " +
                    str(total_novety[i] / 50000 if total_files else 0) + "\n")
            f.write("\n")


if __name__ == "__main__":
    main()
