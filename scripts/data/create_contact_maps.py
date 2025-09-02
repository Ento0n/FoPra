import numpy as np
import os
from Bio.PDB import PDBParser, is_aa
import argparse
import pandas as pd
from tqdm import tqdm

def pdb_to_contact_matrix(pdb_path, threshold=8.0):
    """Return inter-chain CB contact map (0/1) for a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", str(pdb_path))
    model = list(structure)[0]

    coords, chains = [], []

    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            if res.get_resname() == "GLY" and "CA" in res:   # Glycine has no CB
                atom = res["CA"]
            elif "CB" in res:
                atom = res["CB"]
            elif "CA" in res:  # fallback if CB missing
                atom = res["CA"]
            else:
                continue
            coords.append(atom.coord)
            chains.append(chain.id)

    coords = np.array(coords, dtype=np.float32)
    chains = np.array(chains)

    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))

    # Contact map: different chains + <= threshold
    contact_map = ((chains[:, None] != chains[None, :]) & (dist <= threshold)).astype(np.uint8)

    return contact_map

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create contact maps from PDB files")
    parser.add_argument('--path', type=str, default='/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit', help='Base path for dataset splits')
    parser.add_argument('--pdb_path', type=str, default='/nfs/scratch/pinder/pinder/2024-02/pdbs', help='Path for PDB files')
    args = parser.parse_args()

    # Extract unique IDs
    train_df = pd.read_csv(os.path.join(args.path, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.path, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.path, "test.csv"))

    ids = pd.concat([train_df['entry'], val_df['entry'], test_df['entry']]).unique()

    # Create folder for contact matrices if not existent
    os.makedirs(os.path.join(args.path, "contact_maps"), exist_ok=True)

    # create contact maps for collected ids
    for id in tqdm(ids):
        pdb_file = os.path.join(args.pdb_path, f"{id}.pdb")

        C = pdb_to_contact_matrix(pdb_file)
        print(C.shape, C.sum())

        # Save as compressed npz
        np.savez_compressed(os.path.join(args.path, "contact_maps", f"{id}.npz"), contact_map=C)