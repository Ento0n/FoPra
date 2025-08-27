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

os.environ['PINDER_BASE_DIR'] = '/nfs/scratch/pinder'
os.environ['MPLCONFIGDIR'] = '/nfs/scratch/pinder/negative_dataset'
os.environ['HF_HOME'] = '/nfs/scratch/pinder/negative_dataset/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/nfs/scratch/pinder/negative_dataset/cache/huggingface'


# 1. Spin up the ESM client
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
login(token="")
client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)

# 2. Function to save per-sequence embeddings
def save_seq_embeddings(seqs, out_dir):
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
        # skip if already exists
        if os.path.exists(path):
            continue

        # encode each sequence individually
        with torch.no_grad():
            protein_tensor = client.encode(prot)
            result = client.forward_and_sample(
                protein_tensor,
                SamplingConfig(
                    return_per_residue_embeddings=True,
                    return_mean_embedding=False,
                ),
            ).per_residue_embedding

        # save each
        torch.save({"sequence": prot.sequence, "embedding": result.cpu()},
                os.path.join(out_dir, f"{key}.pt"))
            
        # free memory
        del prot, protein_tensor, result
        gc.collect() # delete cache


# 3. Generate embeddings for receptor and ligand sequences
df = pd.concat([
    pd.read_csv("/nfs/scratch/pinder/negative_dataset/datasets/train_dataset.csv"),
    pd.read_csv("/nfs/scratch/pinder/negative_dataset/datasets/val_dataset.csv"),
    pd.read_csv("/nfs/scratch/pinder/negative_dataset/datasets/test_dataset.csv")
], ignore_index=True)

# Only consider sequences where label==1
unique_rec = df[df['label'] == 1]['receptor_seq'].unique()
unique_lig = df[df['label'] == 1]['ligand_seq'].unique()

print("Generating embeddings for receptor sequences...")
save_seq_embeddings(
    unique_rec,
    "/nfs/scratch/pinder/negative_dataset/embeddings/sequence/ESM3/residue"
)
print("Generating embeddings for ligand sequences...")
save_seq_embeddings(
    unique_lig,
    "/nfs/scratch/pinder/negative_dataset/embeddings/sequence/ESM3/residue"
)