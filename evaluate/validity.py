
import os
import sys
from contextlib import contextmanager

from tqdm.contrib.concurrent import process_map
from rdkit import Chem
from rdkit.Chem import AllChem

path = "/data/tzeshinchen/research/gpt2/output/new/"

output_dir = "/data/tzeshinchen/research/evaluate/num/"
invalid_dir = "/data/tzeshinchen/research/evaluate/invalid/"
# if single file
Filename = "overfitting.txt"


output_dir = output_dir + Filename
invalid_dir = invalid_dir + Filename


@contextmanager
def suppress_rdkit_logging():

    rdLogger = Chem.rdBase.DisableLog('rdApp.error')
    try:
        yield
    finally:
        Chem.rdBase.EnableLog('rdApp.error')


def process_file(filename):
    valid_count = 0
    invalid_count = 0
    invalid_smiles = []
    with open(os.path.join(path, filename), 'r') as f:
        smiles_lines = f.read().splitlines()

    for smiles in smiles_lines:
        with suppress_rdkit_logging():
            if not smiles.strip():
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol and mol.GetNumAtoms() > 0:
                mol_with_h = Chem.AddHs(mol)
                try:
                    AllChem.EmbedMolecule(mol_with_h, AllChem.ETKDG())
                    mol_with_h = Chem.RemoveHs(mol_with_h)
                    valid_count += 1
                except ValueError:
                    invalid_count += 1
                    invalid_smiles.append(smiles)

            else:
                invalid_count += 1
                invalid_smiles.append(smiles)

    # Save invalid smiles to a file
    with open(invalid_dir, 'w') as f:
        f.write("Invalid SMILES:\n")
        for smiles in invalid_smiles:
            f.write(smiles + "\n")

    return valid_count, invalid_count


def main():

    multi_process = sys.argv[1] if len(sys.argv) > 1 else False

    dir_list = os.listdir(path)
    total = 10000

    if (multi_process):
        for i in dir_list:
            total_files = len(dir_list)
            results = process_map(process_file, dir_list,
                                  max_workers=os.cpu_count(), chunksize=1)

            total_valid = [result[0] for result in results]
            total_invalid = [result[1] for result in results]

    else:
        total_files = 1
        results = process_file(
            filename=path+Filename)

    with open(output_dir, "w") as f:
        for i in range(total_files):
            if (multi_process):
                f.write(f"File: {dir_list[i]}\n")
            else:
                f.write(f"File: {Filename}\n")

            f.write(f"Total valid SMILES: {total_valid[i]}\n")
            f.write(f"Total invalid SMILES: {total_invalid[i]}\n")
            f.write(f"Overall validation rate: {total_valid[i] / total}\n\n")


if __name__ == "__main__":
    main()
