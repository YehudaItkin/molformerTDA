import numpy as np
from rdkit import Chem

from full_attention_rotary.get_attention_map_full import get_full_attention


def get_bonds_mat(seq):
    """
        this method extracts the bond matrix for a sequenc eusing RDKIT
    """
    mol = Chem.MolFromSmiles(seq)
    bonds = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in mol.GetBonds()]
    bonds_mat = np.zeros((len(mol.GetAtoms()), len(mol.GetAtoms())))

    for ele in bonds:
        bonds_mat[ele[0], ele[1]] = 1
        bonds_mat[ele[1], ele[0]] = 1

    bond_tokens = []

    for atom in mol.GetAtoms():
        bond_tokens.append(atom.GetSymbol())

    return bonds_mat, bond_tokens


def filter_unwanted_tokens(tensor, tokens, ignore_tokens):
    wanted_idx, wanted_tokens = [], []
    ignore_tokens = set(ignore_tokens)
    for idx, token in enumerate(tokens):
        if token in ignore_tokens:
            continue
        else:
            wanted_idx.append(idx)
            wanted_tokens.append(token)

    return tensor[wanted_idx, :][:, wanted_idx], wanted_tokens


def main():
    sequences = ["CC1(C)C(C)(O)C1(C)O", "CC(O)C(C)(O)C(N)=O", "CC(C)C(C)(C)O"]  # gdb_62509, gdb_58097, gdb_1105

    # CHANGE THIS SEQ_IDX To 0,1,2 TO ANALYSIS FOR RESPECTIVE SEQUENCE
    seq_idx = 0
    sequence = sequences[seq_idx]

    attentions, tokens = get_full_attention(sequence)


if __name__ == "__main__":
    main()
