import os
import sys

from matplotlib import pyplot as plt

os.environ['PINDER_BASE_DIR'] = '/nfs/scratch/pinder'
os.environ['MPLCONFIGDIR'] = '/nfs/scratch/pinder/negative_dataset'

from pinder.core import get_pinder_location, get_index, PinderSystem
import random
import pandas as pd
from collections import Counter
from tqdm import tqdm
import argparse


# Remove redundant entries from the dataset
def remove_redundant_entries(index: pd.DataFrame) -> pd.DataFrame:
    """
    Remove redundant entries from the dataset by simplifying chain IDs and dropping duplicates.
    -> Keep only 1 chain conformation per entry

    - Simplifies the 'id' column by removing digits following letters in chain IDs.
    - Drops duplicate entries based on the simplified ID.
    - Removes the helper column before returning.

    Args:
        index (pd.DataFrame): DataFrame with at least an 'id' column.

    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """

    index['id_simple'] = index['id'].str.replace(r'([A-Za-z])\d+', r'\1', regex=True)

    index = index.drop_duplicates(subset='id_simple')

    index = index.drop(columns=['id_simple'])

    return index

def extract_info(df:pd.DataFrame) -> pd.DataFrame:
    """
    Extract sequence and file path information from PinderSystem entries.

    For each entry in the DataFrame, extracts:
      - split
      - receptor and ligand sequences
      - receptor and ligand file paths
      - assigns label 1 (positive pair)

    Skips entries with receptor or ligand sequences longer than 2500 residues.

    Args:
        df (pd.DataFrame): DataFrame with an 'id' column.

    Returns:
        pd.DataFrame: DataFrame with extracted information for each entry.
    """

    records = []
    print("Extracting sequences & paths from Pinder entries... \n")

    flag = False
    for entry in tqdm(df['id']):

        ps = PinderSystem(entry)
        if len(ps.holo_receptor.sequence) > 2500 or len(ps.holo_ligand.sequence) > 2500:
            print(f"Skipping {entry} due to long sequence length.")
            flag = True
            continue

        records.append({
            'entry': entry,
            'split': ps.entry.split,
            'receptor_seq': ps.holo_receptor.sequence,
            'ligand_seq': ps.holo_ligand.sequence,
            'receptor_path': ps.holo_receptor.filepath,
            'ligand_path': ps.holo_ligand.filepath,
            'label': 1,  # Label is always 1 for positive pairs
        })

    # After info to skipping entries was printed, skip a line
    if flag:
        print("\n")

    return pd.DataFrame(records)

def deleak_by_cdhit(train: pd.DataFrame, test: pd.DataFrame, path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove too-similar sequences from train and test splits using CD-HIT-2D results.

    Applies remove_similar_sequences to train and test splits to ensure no similar sequences
    (as determined by CD-HIT-2D) are shared between splits.

    Args:
        train (pd.DataFrame): Training split DataFrame.
        test (pd.DataFrame): Test split DataFrame.
        path (str): Master path for datasets.

    Returns:
        tuple: (train, test) DataFrames after deleaking.
    """
    
    print("Removing too similar sequences determined by CD-HIT-2D!")
    print(f"Using CD-HIT-2D results from {path} \n")
    print(f"Size of train split before CD-HIT deleaking: {len(train)}")
    print(f"Size of val split before CD-HIT deleaking: {len(val)}")
    print(f"Size of test split before CD-HIT deleaking: {len(test)} \n")
    train = remove_similar_sequences(train, 'train', 'val', path)
    train = remove_similar_sequences(train, 'train', 'test', path)
    test = remove_similar_sequences(test, 'test', 'val', path)
    print(f"Size of train split after CD-HIT deleaking: {len(train)}")
    print(f"Size of val split after CD-HIT deleaking: {len(val)}")
    print(f"Size of test split after CD-HIT deleaking: {len(test)} \n")

    return train, test

def remove_similar_sequences(df: pd.DataFrame, df_split: str, other_split: str, path: str) -> pd.DataFrame:
    """
    Remove entries from a split whose receptor or ligand sequences are too similar to those in another split,
    based on CD-HIT-2D results.

    Args:
        df (pd.DataFrame): DataFrame for the split to filter.
        df_split (str): Name of the current split (e.g., 'train').
        other_split (str): Name of the other split compared against by CD-HIT-2D (e.g., 'val' or 'test').
        path (str): Master path for datasets.

    Returns:
        pd.DataFrame: Filtered DataFrame with only allowed sequences.
    """

    # read not too similar sequence
    allowed_seqs = set()
    with open(os.path.join(path, f"{other_split}_{df_split}.out"), "r") as f:
        for line in f:
            if line.startswith(">"):
                allowed_seqs.add(next(f).strip())

    # filter allowed sequences by sequences present
    unique_seqs = pd.concat([df['receptor_seq'], df['ligand_seq']]).unique()
    allowed_seqs = allowed_seqs.intersection(set(unique_seqs))
    
    # give info via cmd line print
    print(f"Removing similar sequences to {other_split} from {df_split} \n")
    print(f"Unique sequences in {df_split}: {len(unique_seqs)}")
    print(f"Number of allowed sequences in {df_split}: {len(allowed_seqs)}")
    print(f"Remove {len(unique_seqs) - len(allowed_seqs)} sequences! \n")

    # only keep entries where receptor and ligand sequences are in allowed sequences
    df = df[df['receptor_seq'].isin(allowed_seqs) & df['ligand_seq'].isin(allowed_seqs)]

    return df

def sample_negatives(df: pd.DataFrame, split: str, n_samples: int, path: bool = True, self_interactions: bool = False) -> pd.DataFrame:
    """
    Sample negative pairs for a given split, matching the sequence frequency distribution of positives.

    Randomly pairs receptor and ligand sequences (not present as positives) with probability proportional
    to their frequency in the positive set.

    Args:
        df (pd.DataFrame): DataFrame of positive pairs for the split.
        split (str): Name of the split.
        n_samples (int): Number of negative samples to generate.

    Returns:
        pd.DataFrame: DataFrame of negative pairs.
    """

    # all positives
    positives = set(zip(df['receptor_seq'], df['ligand_seq']))

    # receptor
    rec_counter = Counter(df['receptor_seq'])
    rec_seqs    = list(rec_counter.keys())
    rec_weights = [rec_counter[s] for s in rec_seqs]

    # ligand
    lig_counter = Counter(df['ligand_seq'])
    lig_seqs    = list(lig_counter.keys())
    lig_weights = [lig_counter[s] for s in lig_seqs]

    # Add as many self interactions as possible (depends on how many sequences are not already self interacting in positives)
    neg_records = []
    if self_interactions:
        self_interacting_df = df[df['receptor_seq'] == df['ligand_seq']]
        n_self_interactions = len(self_interacting_df)

        # sequences not self interacting
        self_interacting_seqs = set(self_interacting_df['receptor_seq'])
        rec_seqs_non_self = [s for s in rec_seqs if s not in self_interacting_seqs]
        lig_seqs_non_self = [s for s in lig_seqs if s not in self_interacting_seqs]
        seqs_non_self = rec_seqs_non_self + lig_seqs_non_self

        if n_self_interactions < len(seqs_non_self):
            seqs_non_self = seqs_non_self[:n_self_interactions]
        
        print(f"Adding {len(seqs_non_self)} self-interactions as negatives in split {split}.")
        print(f"# of self interactions in positives: {n_self_interactions}\n")
        
        for seq in seqs_non_self:
            if path:
                rec_path_obj = df.loc[df['receptor_seq']==seq, 'receptor_path']
                lig_path_obj = df.loc[df['ligand_seq']==seq,   'ligand_path']
                if not rec_path_obj.empty:
                    rec_path = rec_path_obj.iat[0]
                    lig_path = rec_path_obj.iat[0]
                elif not lig_path_obj.empty:
                    rec_path = lig_path_obj.iat[0]
                    lig_path = lig_path_obj.iat[0]
                else:
                    print(f"Warning: Could not find path for self-interacting sequence {seq} in split {split}. Paths will be set to None.")
                    rec_path = None
                    lig_path = None
                
                neg_records.append({
                    'split': split,
                    'receptor_seq': seq,
                    'ligand_seq':  seq,
                    'receptor_path': rec_path,
                    'ligand_path':  lig_path,
                    'label': 0
                })
            else:
                neg_records.append({
                    'split': split,
                    'receptor_seq': seq,
                    'ligand_seq':  seq,
                    'label': 0
                })


    while len(neg_records) < n_samples:
        rec_seq = random.choices(rec_seqs, weights=rec_weights, k=1)[0]
        lig_seq = random.choices(lig_seqs, weights=lig_weights, k=1)[0]
        
        if (rec_seq, lig_seq) in positives:
            continue

        if path:
            rec_path = df.loc[df['receptor_seq']==rec_seq, 'receptor_path'].iat[0]
            lig_path = df.loc[df['ligand_seq']==lig_seq,   'ligand_path'].iat[0]
            neg_records.append({
                'split': split,
                'receptor_seq': rec_seq,
                'ligand_seq':  lig_seq,
                'receptor_path': rec_path,
                'ligand_path':  lig_path,
                'label': 0
            })
        else:
            neg_records.append({
                'split': split,
                'receptor_seq': rec_seq,
                'ligand_seq':  lig_seq,
                'label': 0
            })

    return pd.DataFrame(neg_records)

def deleak_by_uniprot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove entries with UniProt IDs that are shared between train, val, and test splits.

    Ensures that each UniProt ID appears in at most one split, preferring to keep it in val/test
    over train, and in test over val.

    Args:
        df (pd.DataFrame): DataFrame with 'split', 'uniprot_L', and 'uniprot_R' columns.

    Returns:
        pd.DataFrame: Deleaked DataFrame.
    """

    # Filter for valid splits
    df_valid = df[df['split'].isin({"train", "val", "test"})]

    # Melt to get all uniprot IDs in a single column, keeping split info
    uniprot_melted = pd.melt(
        df_valid,
        id_vars=['split'],
        value_vars=['uniprot_L', 'uniprot_R'],
        var_name='uniprot_side',
        value_name='uniprot_id'
    )

    # Drop rows with missing uniprot_id
    uniprot_melted = uniprot_melted.dropna(subset=['uniprot_id'])

    # For each uniprot_id, count how many unique splits it appears in
    split_counts = uniprot_melted.groupby('uniprot_id')['split'].nunique()

    # Find uniprot_ids that appear in more than one split
    shared_uniprot_ids = split_counts[split_counts > 1].index.tolist()

    # Remove rows so that each uniprot_id appears in at most one of train/val/test, preferring to keep in val/test

    # 1. For each shared uniprot_id, get all rows in train/val/test
    mask_shared = uniprot_melted['uniprot_id'].isin(shared_uniprot_ids)
    shared_rows = uniprot_melted[mask_shared]

    # 2. For each shared uniprot_id, determine which splits it appears in
    uniprot_to_splits = shared_rows.groupby('uniprot_id')['split'].unique()

    # 3. Build a set of (uniprot_id, split) pairs to remove from train
    to_remove = set()
    for uniprot_id, splits in uniprot_to_splits.items():
        if 'test' in splits:
            # if in test, drop from train and val
            if 'train' in splits:
                to_remove.add(('train', uniprot_id))
            if 'val' in splits:
                to_remove.add(('val', uniprot_id))
        elif 'val' in splits and 'train' in splits:
            # if in val (but not test) and train, drop from train
            to_remove.add(('train', uniprot_id))

    # 4. Remove from df_valid all rows where uniprot_L or uniprot_R is in a to_remove uniprot_id
    def should_remove(row):
        return ((row['split'], row['uniprot_L']) in to_remove
                or (row['split'], row['uniprot_R']) in to_remove)

    mask = df_valid.apply(should_remove, axis=1)
    df_valid_deleaked = df_valid[~mask].copy()

    return df_valid_deleaked


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create positive and negative datasets by split")
    parser.add_argument('--out_path', type=str, default=None, required=True, help='Output path for dataset splits')

    parser.add_argument('--exist_path', type=str, default=None, required=False, help='Path to existing dataset splits to load from')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--deleak_uniprot', action='store_true', help='Deleak with by uniprot ids')
    group.add_argument('--deleak_cdhit', action='store_true', help='Deleak with by cd-hit-2d created fasta files')
    parser.add_argument('--fasta_splits_path', type=str, help='Path to fasta files for cd-hit-2d deleaking')

    parser.add_argument('--self_interactions', action='store_true', help='Consider self-interactions when sampling negatives')

    args = parser.parse_args()
    
    if (args.deleak_uniprot or args.deleak_cdhit) and not args.exist_path:
        parser.error("Deleaking requires an existing dataset path to load from. Please provide --exist_path.")
    
    if args.deleak_cdhit and not args.fasta_splits_path:
        parser.error("CD-HIT deleaking requires a path to fasta files. Please provide --fasta_splits_path.")

    # Load save point if exist_path is given
    if args.exist_path:
        try:
            train = pd.read_csv(os.path.join(args.exist_path, "train.csv"))
            val   = pd.read_csv(os.path.join(args.exist_path, "val.csv"))
            test  = pd.read_csv(os.path.join(args.exist_path, "test.csv"))
        except FileNotFoundError:
            print(f"Error: Could not find dataset splits in the provided exist_path: {args.exist_path}")
            sys.exit(1)
        else:
            print(f"Using existing Pinder dataset splits from {args.exist_path} \n")

        # consider only positives for deleaking
        train = train[train["label"] == 1]
        val   = val[val["label"] == 1]
        test  = test[test["label"] == 1]

        # only 1 index dataframe used for deleaking
        index = pd.concat([train, val, test], ignore_index=True)

        if args.deleak_uniprot:
            print("######### Deleaking dataset by Uniprot IDs #########\n")
            index = deleak_by_uniprot(index)

            # split back into train, val, test
            train = index[index["split"] == "train"]
            val = index[index["split"] == "val"]
            test = index[index["split"] == "test"]
        
        if args.deleak_cdhit:
            print("######### Deleaking dataset by CD-HIT #########\n")
            train, test = deleak_by_cdhit(train, test, args.fasta_splits_path)

    # If no exist_path is given, create dataset from scratch
    else:
        print("######### Create Pinder dataset splits from scratch #########\n")

        # Test pinder location
        print(f"pinder location: {get_pinder_location()}")
    
        # Get the index of Pinder systems
        index = get_index().copy()
    
        index = remove_redundant_entries(index)
    
        # extract info for all entries
        index = extract_info(index)

        # split into train, val, test
        train = index[index["split"] == "train"]
        val   = index[index["split"] == "val"]
        test  = index[index["split"] == "test"]
    
    # Positives are now ready, sample negatives and save splits
    print("######### Sample negatives and save dataset splits #########\n")

    # Print sizes of splits
    print(f"Size of train split: {len(train)}")
    print(f"Size of val split: {len(val)}")
    print(f"Size of test split: {len(test)}\n")

    # Sample negatives for each split
    neg_train = sample_negatives(train, "train", len(train))
    neg_val   = sample_negatives(val, "val", len(val))
    neg_test  = sample_negatives(test, "test", len(test))

    # Concatenate positive and negative samples
    train = pd.concat([train, neg_train], ignore_index=True)
    val   = pd.concat([val,   neg_val],   ignore_index=True)
    test  = pd.concat([test,  neg_test],  ignore_index=True)

    # Save splits
    print("######### Save dataset splits #########\n")
    print(f"Save dataset splits to {args.out_path}")
    train.to_csv(os.path.join(args.out_path, 'train.csv'), index=False)
    val.to_csv(os.path.join(args.out_path, 'val.csv'), index=False)
    test.to_csv(os.path.join(args.out_path, 'test.csv'), index=False)










