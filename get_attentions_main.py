import glob
import os
from argparse import Namespace

import pandas as pd
import torch

from get_full_attention import get_full_attention_matrices
from get_linear_attention import get_linear_attention_matrices

IGNORE_TOKENS = ["(", ")", "=", ""]
IGNORE_TOKENS += ["1", "2", ]
IGNORE_TOKENS += ["<eos>", "<bos>"]


def work_on_dataset(args):
    dataset_path = args.dataset_path
    basename = dataset_path.split("/")[-1]
    output_dir = os.path.join("output", basename)
    os.makedirs(output_dir, exist_ok=True)
    full_output = []
    for fname in glob.glob(os.path.join(dataset_path, "*.csv")):
        print(f"working on {fname}")
        dataset_type = os.path.basename(fname).split(".")[0]
        df = pd.read_csv(fname)
        for row in df.iterrows():
            smiles = row[1]['smiles']
            name = row[1]['smiles']
            print(f"working on {name} ({smiles})")
            if args.attention_type.lower() == "linear":
                output = get_linear_attention_matrices(smiles, args.ignore_tokens)
            elif args.attention_type.lower() == "full":
                output = get_full_attention_matrices(smiles, args.ignore_tokens)
            else:
                raise Exception(f"Unknown attention type {args.attention_type}")
            output['name'] = name
            output['smiles'] = smiles
            output['ignore_tokens'] = args.ignore_tokens
            full_output.append(output)

        output_filename = f"{dataset_type} + {args.attention_type.lower()}"
        if args.ignore_tokens:
            output_filename += "_ignore_tokens"
        else:
            output_filename += "_all_tokens"
        output_filename += ".pth"
        torch.save(output_dir, output_filename)
def main():
    args = Namespace()
    args.ignore_tokens = IGNORE_TOKENS
    args.attention_type = "linear"
    args.dataset_path = "./data/data/esol"
    work_on_dataset(args)
