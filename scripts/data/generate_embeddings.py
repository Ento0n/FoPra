import os
import hashlib
import torch
import pandas as pd
from tqdm import tqdm
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.models.esm3 import ESM3
from esm.utils.constants.models import ESM3_OPEN_SMALL
from huggingface_hub import login
import gc
import argparse

os.environ['PINDER_BASE_DIR'] = '/nfs/scratch/pinder'
os.environ['MPLCONFIGDIR'] = '/nfs/scratch/pinder/negative_dataset'
os.environ['HF_HOME'] = '/nfs/scratch/pinder/negative_dataset/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/nfs/scratch/pinder/negative_dataset/cache/huggingface'


# 2. Function to save per-sequence embeddings
def save_seq_embeddings(seqs, residue, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    max_seq_len = 0
    for seq in tqdm(seqs):
        # update max sequence length
        max_seq_len = max(max_seq_len, len(seq))
        print(f"Max sequence length: {max_seq_len}")

        # generate ESM protein object
        prot = ESMProtein(sequence=seq)

        # skip already‐saved ones
        key = hashlib.md5(prot.sequence.encode()).hexdigest()
        path = os.path.join(out_dir, f"{key}.pt")
        if os.path.exists(path):
            continue

        # encode each sequence individually
        with torch.no_grad():
            protein_tensor = client.encode(prot)
            if residue:
                result = client.forward_and_sample(
                    protein_tensor,
                    SamplingConfig(
                        return_per_residue_embeddings=True,
                        return_mean_embedding=False,
                    ),
                ).per_residue_embedding
            else:
                result = client.forward_and_sample(
                    protein_tensor,
                    SamplingConfig(
                        return_per_residue_embeddings=False,
                        return_mean_embedding=True,
                    ),
                ).mean_embedding

        # save each
        torch.save({"sequence": prot.sequence, "embedding": result.cpu()},
                os.path.join(out_dir, f"{key}.pt"))
            
        # free memory
        del prot, protein_tensor, result
        gc.collect() # delete cache


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Embeddings for Protein information")
    parser.add_argument('--token', type=str, default=None, required=True, help='Token for huggingface login')
    parser.add_argument('--path', type=str, default=None, required=True, help='Path to the dataset CSV file')

    args = parser.parse_args()

    # 1. Spin up the ESM client
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    login(token=args.token)
    client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)

    # whether to save residue‐level embeddings or mean embeddings
    residue = False

    # 3. Generate embeddings for receptor and ligand sequences
    print(f"Loading dataset from {args.path}...\n")
    df = pd.concat([
        pd.read_csv(os.path.join(args.path, "train.csv")),
        pd.read_csv(os.path.join(args.path, "val.csv")),
        pd.read_csv(os.path.join(args.path, "test.csv"))
    ], ignore_index=True)

    # Only consider sequences where label==1
    unique_rec = df['receptor_seq'].unique()
    unique_lig = df['ligand_seq'].unique()

    if residue:
        out_path = "/nfs/scratch/pinder/negative_dataset/my_repository/embeddings/sequence/ESM3/residue"
    else:
        out_path = "/nfs/scratch/pinder/negative_dataset/my_repository/embeddings/sequence/ESM3/mean"
    
    print(f"Saving embeddings to {out_path}...\n")

    print("Generating embeddings for receptor sequences...\n")
    save_seq_embeddings(
        unique_rec,
        residue,
        out_path
    )
    print("Generating embeddings for ligand sequences...\n")
    save_seq_embeddings(
        unique_lig,
        residue,
        out_path
    )