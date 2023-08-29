import glob
import os
from argparse import Namespace

import pandas as pd
import torch

from linear_attention import get_linear_attention_matrices

IGNORE_TOKENS = ["(", ")", "=", ""]
IGNORE_TOKENS += ["1", "2", ]
IGNORE_TOKENS += ["<eos>", "<bos>"]


def work_on_dataset(args):
    dataset_path = args.dataset_path
    basename = dataset_path.split("/")[-1]
    output_dir = os.path.join(args.outdir, basename)
    os.makedirs(output_dir, exist_ok=True)

    for fname in glob.glob(os.path.join(dataset_path, "*.csv")):
        full_output = []
        dataset_type = os.path.basename(fname).split(".")[0]
        df = pd.read_csv(fname)
        print(f"working on {fname} with len = {len(df)}")
        for idx, row in enumerate(df.iterrows()):
            smiles = row[1]['smiles']
            name = row[1].get('name', smiles)
            print(f"working on {idx}: {name} ({smiles})")
            if args.attention_type.lower() == "linear":
                output = get_linear_attention_matrices(smiles, args.ignore_tokens)
            else:
                raise Exception(f"Unknown attention type {args.attention_type}")
            output['name'] = name
            output['smiles'] = smiles
            output['ignore_tokens'] = args.ignore_tokens
            full_output.append(output)
        assert len(full_output) == len(df)
        output_filename = f"{dataset_type}_{args.attention_type.lower()}"
        if args.ignore_tokens:
            output_filename += "_ignore_tokens"
        else:
            output_filename += "_all_tokens"
        output_filename += ".pth"
        output_filename = os.path.join(output_dir, output_filename)
        print(f"Writing to {output_filename}, len(full_output)={len(full_output)})")
        torch.save(full_output, output_filename)


def main():
    args = Namespace()
    args.outdir = "output"
    args.ignore_tokens = IGNORE_TOKENS
    args.attention_type = "linear"
    for d_path in ["./data/data/esol", "./data/data/freesolv", "./data/data/lipo"]:
        args.dataset_path = d_path
        for t in [[], IGNORE_TOKENS]:
            args.ignore_tokens = t
            work_on_dataset(args)


if __name__ == "__main__":
    main()
