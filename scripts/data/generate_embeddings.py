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

    for seq in seqs:
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

def save_structure_embeddings(df, residue, out_dir):
    for _, row in df.iterrows():
        # generate ESM protein object
        rec_chain = row["entry"].split("--")[0].split("_")[-2]
        lig_chain = row["entry"].split("--")[1].split("_")[-2]

        # print(f"Processing entry {row['entry']} with receptor chain {rec_chain} and ligand chain {lig_chain}")
        # chain extraction works!

        rec_prot = ESMProtein.from_pdb(row['receptor_path'], chain_id="detect")
        lig_prot = ESMProtein.from_pdb(row['ligand_path'], chain_id="detect")

        if rec_prot.coordinates is None or lig_prot.coordinates is None:
            print(f"Skipping entry {row['entry']} due to missing coordinates.")
            continue

        # Check if the residues are more than 2
        if len(rec_prot.sequence) <= 2 or len(lig_prot.sequence) <= 2:
            print(f"Skipping entry {row['entry']} due to insufficient residues.")
            continue

        # skip already‐saved ones
        rec_key = hashlib.md5(rec_prot.sequence.encode()).hexdigest()
        lig_key = hashlib.md5(lig_prot.sequence.encode()).hexdigest()
        rec_path = os.path.join(out_dir, f"{rec_key}.pt")
        lig_path = os.path.join(out_dir, f"{lig_key}.pt")
        lig_exists = os.path.exists(lig_path)
        rec_exists = os.path.exists(rec_path)
        if rec_exists and lig_exists:
            continue

        # encode each sequence individually
        with torch.no_grad():
            if not rec_exists:
                rec_tensor = client.encode(rec_prot)
                if residue:
                    rec_result = client.forward_and_sample(
                        rec_tensor,
                        SamplingConfig(
                            return_per_residue_embeddings=True,
                            return_mean_embedding=False,
                        ),
                    ).per_residue_embedding
                else:
                    rec_result = client.forward_and_sample(
                        rec_tensor,
                        SamplingConfig(
                            return_per_residue_embeddings=False,
                            return_mean_embedding=True,
                        ),
                    ).mean_embedding

            if not lig_exists:
                lig_tensor = client.encode(lig_prot)
                if residue:
                    lig_result = client.forward_and_sample(
                        lig_tensor,
                        SamplingConfig(
                            return_per_residue_embeddings=True,
                            return_mean_embedding=False,
                        ),
                    ).per_residue_embedding
                else:
                    lig_result = client.forward_and_sample(
                        lig_tensor,
                        SamplingConfig(
                            return_per_residue_embeddings=False,
                            return_mean_embedding=True,
                        ),
                    ).mean_embedding
        
        # save each
        if not rec_exists:
            torch.save({"sequence": rec_prot.sequence, "embedding": rec_result.cpu()},
                    os.path.join(out_dir, f"{rec_key}.pt"))
        if not lig_exists:
            torch.save({"sequence": lig_prot.sequence, "embedding": lig_result.cpu()},
                    os.path.join(out_dir, f"{lig_key}.pt"))


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Embeddings for Protein information")
    parser.add_argument('--token', type=str, default=None, required=True, help='Token for huggingface login')
    parser.add_argument('--residue', action='store_true', help='Whether to generate per-residue embeddings (default is mean embeddings)')
    parser.add_argument('--path', type=str, default=None, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--out_path', type=str, default=None, required=True, help='Output path to save embeddings')
    parser.add_argument('--structure', action='store_true', help='Whether to generate structure-based embeddings instead of sequence-based')

    args = parser.parse_args()

    # 1. Spin up the ESM client
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    login(token=args.token)
    client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)

    # 3. Generate embeddings for receptor and ligand sequences
    print(f"Loading dataset from {args.path}...\n")
    df = pd.concat([
        pd.read_csv(os.path.join(args.path, "train.csv")),
        pd.read_csv(os.path.join(args.path, "val.csv")),
        pd.read_csv(os.path.join(args.path, "test.csv"))
    ], ignore_index=True)

    if args.structure:
        print(f"Saving structure&sequence-based embeddings to {args.out_path}...\n")
        save_structure_embeddings(
            df,
            args.residue,
            args.out_path
        )
    else:
        # Only consider sequences where label==1
        unique_rec = df['receptor_seq'].unique()
        unique_lig = df['ligand_seq'].unique()

        print(f"Saving sequence embeddings to {args.out_path}...\n")

        print("Generating embeddings for receptor sequences...\n")
        save_seq_embeddings(
            unique_rec,
            args.residue,
            args.out_path
        )
        print("Generating embeddings for ligand sequences...\n")
        save_seq_embeddings(
            unique_lig,
            args.residue,
            args.out_path
        )