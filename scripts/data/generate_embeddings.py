import os

# Set cache/config dirs before importing heavy libraries so they pick up writable locations.
os.environ['PINDER_BASE_DIR'] = '/nfs/scratch/pinder'
os.environ['MPLCONFIGDIR'] = '/nfs/scratch/pinder/negative_dataset'
os.environ['HF_HOME'] = '/nfs/scratch/pinder/negative_dataset/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/nfs/scratch/pinder/negative_dataset/cache/huggingface'
os.environ['TORCH_HOME'] = '/nfs/scratch/pinder/negative_dataset/cache/torch'
os.environ['XDG_CACHE_HOME'] = '/nfs/scratch/pinder/negative_dataset/cache'

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
import re
import tempfile


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

def preprocess_pdb_content(pdb_path):
    """Convert HETATM to ATOM and rename D-amino acids on the fly"""
    with open(pdb_path, 'r') as f:
        content = f.read()
    
    # Convert HETATM to ATOM
    content = re.sub(r'^HETATM', 'ATOM  ', content, flags=re.MULTILINE)
    
    # Rename D-amino acids to their L-equivalents
    d_amino_mappings = {
        ' DAL ': ' ALA ',
        ' DCY ': ' CYS ',
        ' DGL ': ' GLU ',
        ' DAR ': ' ARG ',
        ' DHI ': ' HIS ',
        ' DTR ': ' TRP ',
        ' DLE ': ' LEU ',
        ' DAS ': ' ASP ',
        ' DLY ': ' LYS ',
        ' DPH ': ' PHE ',
        ' DTY ': ' TYR ',
        ' DSE ': ' SER ',
        ' DTH ': ' THR ',
        ' DGN ': ' GLN ',
        ' DPR ': ' PRO ',
        ' DVA ': ' VAL ',
        ' DIL ': ' ILE ',
        ' DME ': ' MET '
    }

    # Map non-standard HETATM residues to standard amino acids
    hetatm_mappings = {
        # Modified Cysteines
        ' BB9 ': ' CYS ',  # Your BB9 case
        ' CSD ': ' CYS ',  # S-sulfanylcysteine
        ' CSO ': ' CYS ',  # S-hydroxycysteine
        ' CSX ': ' CYS ',  # S-oxy cysteine
        ' CME ': ' CYS ',  # S-methylcysteine
        ' CSW ': ' CYS ',  # Cysteine-S-dioxide
        ' CYM ': ' CYS ',  # Deprotonated cysteine
        
        # Selenocysteine and Selenomethionine
        ' SEC ': ' CYS ',  # Selenocysteine -> CYS
        ' MSE ': ' MET ',  # Selenomethionine -> MET
        
        # Modified Serines/Threonines
        ' SEP ': ' SER ',  # Phosphoserine
        ' TPO ': ' THR ',  # Phosphothreonine
        ' MOZ ': ' SER ',  # Your MOZ case - likely modified serine
        
        # Modified Tyrosines
        ' PTR ': ' TYR ',  # Phosphotyrosine
        ' TYS ': ' TYR ',  # Sulfonated tyrosine
        
        # Modified Lysines
        ' MLY ': ' LYS ',  # N-methyllysine
        ' M3L ': ' LYS ',  # N-trimethyllysine
        ' ALY ': ' LYS ',  # N-acetyllysine
        
        # Modified Arginines
        ' AGM ': ' ARG ',  # Modified arginine
        
        # Modified Histidines
        ' HID ': ' HIS ',  # Delta-protonated histidine
        ' HIE ': ' HIS ',  # Epsilon-protonated histidine
        ' HIP ': ' HIS ',  # Doubly protonated histidine
        
        # Modified Prolines
        ' HYP ': ' PRO ',  # Hydroxyproline
        
        # Modified Leucines
        ' NLE ': ' LEU ',  # Norleucine
        
        # Modified Phenylalanines
        ' PHD ': ' PHE ',  # Modified phenylalanine
        
        # Modified Glutamates/Aspartates
        ' PCA ': ' GLU ',  # Pyroglutamic acid
        ' CGU ': ' GLU ',  # Gamma-carboxy-glutamic acid
        
        # N-terminal and C-terminal modifications
        ' ACE ': ' ALA ',  # Acetyl group (often treat as ALA or remove)
        ' NH2 ': ' GLY ',  # Amide C-terminus (often treat as GLY or remove)
        
        # Other common modifications
        ' CAS ': ' CYS ',  # S-(dimethylarsenic)cysteine
        ' OCS ': ' CYS ',  # Cysteinesulfonic acid
    }

    # Apply D-amino acid mappings
    for d_res, l_res in d_amino_mappings.items():
        content = content.replace(d_res, l_res)
    
    # Apply HETATM residue mappings
    for hetatm_res, std_res in hetatm_mappings.items():
        content = content.replace(hetatm_res, std_res)
    
    return content

def save_structure_embeddings(df, residue, out_dir):
    sequence_missmatch = {}
    faulty = {}
    for _, row in df.iterrows():
        try:
            # Preprocess PDB files on the fly
            rec_content = preprocess_pdb_content(row['receptor_path'])
            lig_content = preprocess_pdb_content(row['ligand_path'])
            
            # Create temporary files with preprocessed content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as rec_temp:
                rec_temp.write(rec_content)
                rec_temp_path = rec_temp.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as lig_temp:
                lig_temp.write(lig_content)
                lig_temp_path = lig_temp.name
            
            # Load ESMProtein from preprocessed PDB files
            rec_prot = ESMProtein.from_pdb(rec_temp_path, chain_id="detect")
            lig_prot = ESMProtein.from_pdb(lig_temp_path, chain_id="detect")
            
            # Clean up temporary files
            os.unlink(rec_temp_path)
            os.unlink(lig_temp_path)
            
        except Exception as e:
            faulty[row['entry']] = str(e)
            continue
        
        if rec_prot.sequence != row['receptor_seq']:
            sequence_missmatch[row["receptor_seq"]] = f"Receptor sequence mismatch: expected {row['receptor_seq']}, got {rec_prot.sequence}"

        if lig_prot.sequence != row['ligand_seq']:
            sequence_missmatch[row["ligand_seq"]] = f"Ligand sequence mismatch: expected {row['ligand_seq']}, got {lig_prot.sequence}"

        skip_rec = False
        skip_lig = False
        if rec_prot.coordinates is None and lig_prot.coordinates is None:
            faulty[row['entry']] = f"Missing both coordinates for entry {row['entry']}."
            continue
        elif rec_prot.coordinates is None:
            faulty[row['entry']] = f"Missing receptor coordinates for entry {row['entry']}."
            skip_rec = True
        elif lig_prot.coordinates is None:
            faulty[row['entry']] = f"Missing ligand coordinates for entry {row['entry']}."
            skip_lig = True

        # Check if the residues are more than 2
        if len(rec_prot.sequence) <= 2 or len(lig_prot.sequence) <= 2:
            faulty[row['entry']] = f"Skipping entry {row['entry']} due to insufficient residues."
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
            if not rec_exists and not skip_rec:
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

            if not lig_exists and not skip_lig:
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
        if not rec_exists and not skip_rec:
            torch.save({"sequence": rec_prot.sequence, "embedding": rec_result.cpu()},
                    os.path.join(out_dir, f"{rec_key}.pt"))
        if not lig_exists and not skip_lig:
            torch.save({"sequence": lig_prot.sequence, "embedding": lig_result.cpu()},
                    os.path.join(out_dir, f"{lig_key}.pt"))
    
    for entry, error in faulty.items():
        print(f"Entry {entry} skipped due to error: {error}")
    
    for seq, msg in sequence_missmatch.items():
        print(f"Sequence mismatch for sequence {seq}: {msg}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Embeddings for Protein information")
    parser.add_argument('--token', type=str, default=None, required=True, help='Token for huggingface login')
    parser.add_argument('--residue', action='store_true', help='Whether to generate per-residue embeddings (default is mean embeddings)')
    parser.add_argument('--path', type=str, default=None, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--out_path', type=str, default=None, required=True, help='Output path to save embeddings')
    parser.add_argument('--structure', action='store_true', help='Whether to generate structure-based embeddings instead of sequence-based')
    parser.add_argument('--model', type=str, default='esm3', help='Model to use for embedding generation (default: esm3)')

    args = parser.parse_args()

    if args.residue:
        print("Generating per-residue embeddings...\n")
    else:
        print("Generating mean embeddings...\n")

    if args.model.lower() == 'esm3':
        print("Generating embeddings using ESM-3 model...\n")
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

        # Consider only sequences where label==1
        df = df[df["label"] == 1]

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
    
    elif args.model.lower() == 'esm2':
        import esm

        print("Saving sequence embeddings using ESM-2 model...\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()
        target_layer = model.num_layers  # final layer

        print(f"Loading dataset from {args.path}...\n")
        df = pd.concat([
            pd.read_csv(os.path.join(args.path, "train.csv")),
            pd.read_csv(os.path.join(args.path, "val.csv")),
            pd.read_csv(os.path.join(args.path, "test.csv"))
        ], ignore_index=True)
        df = df[df["label"] == 1]

        def save_seq_embeddings_esm2(seqs, residue, out_dir):
            os.makedirs(out_dir, exist_ok=True)

            for seq in tqdm(seqs, desc="Sequences"):
                key = hashlib.md5(seq.encode()).hexdigest()
                path = os.path.join(out_dir, f"{key}.pt")
                if os.path.exists(path):
                    continue

                batch = [("protein", seq)]
                _, _, tokens = batch_converter(batch)
                tokens = tokens.to(device)

                with torch.no_grad():
                    outputs = model(tokens, repr_layers=[target_layer], return_contacts=False)
                    rep = outputs["representations"][target_layer][:, 1:-1, :]  # trim BOS/EOS
                    embedding = rep.squeeze(0) if residue else rep.mean(1)

                torch.save({"sequence": seq, "embedding": embedding.cpu()}, path)

        print(f"Saving sequence embeddings to {args.out_path}...\n")
        save_seq_embeddings_esm2(df['receptor_seq'].unique(), args.residue, args.out_path)
        save_seq_embeddings_esm2(df['ligand_seq'].unique(), args.residue, args.out_path)
        
    else:
        raise ValueError(f"Model {args.model} not supported yet.")
