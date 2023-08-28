import numpy as np
import torch
from rdkit import Chem

from linear_attention_rotary.get_attention_map_full import get_full_attention


def get_bonds_mat(seq):
    """
        this method extracts the bond matrix for a sequenc eusing RDKIT
    """
    mol = Chem.MolFromSmiles(seq)
    bonds = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in mol.GetBonds()]
    bonds_mat = np.zeros((len(mol.GetAtoms()), len(mol.GetAtoms())))

    for ele in bonds:
        if ele[0] >= len(bonds) or ele[1] >= len(bonds):
            continue
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



def get_linear_attention_matrices(sequence, ignore_tokens):
    attentions, tokens = get_full_attention(sequence)

    bonds_mat, gold_tokens = get_bonds_mat(sequence)

    output = {}
    for layer_idx in range(len(attentions)):
        some_layer = attentions[layer_idx].squeeze()
        tensor = torch.mean(some_layer, dim=0)

        tensor = ((tensor + torch.transpose(tensor, 0, 1)) / 2).cpu()
        tensor.fill_diagonal_(0.0)
        tensor = tensor.numpy()

        tensor, filtered_tokens = filter_unwanted_tokens(tensor, tokens, ignore_tokens)
        output[layer_idx] = dict()
        output[layer_idx]['attention'] = tensor
        output[layer_idx]['bonds_mat'] = bonds_mat
        output[layer_idx]['filtered_tokens'] = filtered_tokens
        output[layer_idx]['gold_tokens'] = gold_tokens

    return output


def main():
    sequences = ["CC1(C)C(C)(O)C1(C)O", "CC(O)C(C)(O)C(N)=O", "CC(C)C(C)(C)O"]  # gdb_62509, gdb_58097, gdb_1105

    # CHANGE THIS SEQ_IDX To 0,1,2 TO ANALYSIS FOR RESPECTIVE SEQUENCE
    seq_idx = 0
    sequence = sequences[seq_idx]

    attentions, tokens = get_full_attention(sequence)

    bonds_mat, gold_tokens = get_bonds_mat(sequence)
    # ignore_tokens = []
    ignore_tokens = ["(", ")", "=", ""]
    ignore_tokens += ["1", "2", ]
    ignore_tokens += ["<eos>", "<bos>"]

    output = {}
    for layer_idx in range(len(attentions)):
        some_layer = attentions[layer_idx].squeeze()
        tensor = torch.mean(some_layer, dim=0)

        tensor = ((tensor + torch.transpose(tensor, 0, 1)) / 2).cpu()
        tensor.fill_diagonal_(0.0)
        tensor = tensor.numpy()

        tensor, filtered_tokens = filter_unwanted_tokens(tensor, tokens, ignore_tokens)
        output[layer_idx] = dict()
        output[layer_idx]['attention'] = tensor
        output[layer_idx]['bonds_mat'] = bonds_mat
        output[layer_idx]['filtered_tokens'] = filtered_tokens
        output[layer_idx]['gold_tokens'] = gold_tokens

    torch.save(output, "linear_att.pth")



if __name__ == "__main__":
    main()
