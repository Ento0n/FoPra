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
    print(f"Using CD-HIT-2D results from {os.path.join(args.path, 'fasta')} \n")
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
    with open(os.path.join(path, "fasta", f"{other_split}_{df_split}.out"), "r") as f:
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


def helper_create_fasta_file(df: pd.DataFrame, path: str) -> None:
    """
    Create and save FASTA files for the train, val, and test splits.

    Args:
        train (pd.DataFrame): Training split DataFrame.
        val (pd.DataFrame): Validation split DataFrame.
        test (pd.DataFrame): Test split DataFrame.
        path (str): Master path for datasets.

    Returns:
        None
    """    

    print("Fasta files for the splits train, val & test are created...")
    create_fasta_file(df[df['split'] == 'train'], path, 'train')
    create_fasta_file(df[df['split'] == 'val'],   path, 'val')
    create_fasta_file(df[df['split'] == 'test'],  path, 'test')
    print("Fasta files created succesfully, finish execution! \n")

    # creation of fasta files is a exclusive job
    sys.exit()

def create_fasta_file(df: pd.DataFrame, split: str, path: str) -> None:
    """
    Create a FASTA file for a given split, containing all unique receptor and ligand sequences.

    Args:
        df (pd.DataFrame): DataFrame for the split.
        split (str): Name of the split (e.g., 'train', 'val', 'test').
        path (str): Directory to save the FASTA file.

    Returns:
        None
    """

    path = os.path.join(path, 'fasta')
    os.makedirs(path, exist_ok=True) # Create directory if it doesn't exist
    path = os.path.join(path, f"{split}.fasta")
    all_seqs = list(df['receptor_seq']) + list(df['ligand_seq'])
    all_seqs_unique = set(all_seqs)

    with open(path, 'w') as f:
        for i, seq in enumerate(all_seqs_unique):
            f.write(f">{f"{i}_{split}"}\n{seq}\n")

def sample_negatives(df: pd.DataFrame, split: str, n_samples: int) -> pd.DataFrame:
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

    positives = set(zip(df['receptor_seq'], df['ligand_seq']))
    rec_counter = Counter(df['receptor_seq'])
    rec_seqs    = list(rec_counter.keys())
    rec_weights = [rec_counter[s] for s in rec_seqs]
    lig_counter = Counter(df['ligand_seq'])
    lig_seqs    = list(lig_counter.keys())
    lig_weights = [lig_counter[s] for s in lig_seqs]

    neg_records = []
    while len(neg_records) < n_samples:
        rec_seq = random.choices(rec_seqs, weights=rec_weights, k=1)[0]
        lig_seq = random.choices(lig_seqs, weights=lig_weights, k=1)[0]
        if (rec_seq, lig_seq) in positives:
            continue
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
    parser.add_argument('--train_lim', type=int, default=None, required=False, help='Upper limit for training set size')
    parser.add_argument('--deleak_uniprot', action='store_true', required=False, help='Deleak the dataset by removing duplicates across splits')
    parser.add_argument('--path', type=str, default=None, required=True, help='Base path for datasets')

    # Add mutually exclusive arguments for fasta creation and deleaking
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--create_fasta', action='store_true', help='Create and save a FASTA file of the splits')
    group.add_argument('--deleak_cdhit', action='store_true', help='Deleak with by cd-hit-2d created fasta files')

    args = parser.parse_args()

    # Check if Pinder data without uniprot deleak is already saved as .csv file
    print("######### Loading Pinder dataset #########\n")

    # Somehow some pdbs are not existent and want to be downloaded which results in error since pdb path is not writable
    # -> commented next block to immediatly take uniprot deleaked data

    index = pd.DataFrame()

    # Load save point if exists
    # if os.path.exists(os.path.join(args.path, 'raw', 'all_positives.csv')):
    #     index = pd.read_csv(os.path.join(args.path, 'raw', 'all_positives.csv'))
    #     print("Using existing Pinder dataframe without uniprot deleaking from CSV.")
    # else:
    #     # Test pinder location
    #     print(f"pinder location: {get_pinder_location()}")
    # 
    #     # Get the index of Pinder systems
    #     index = get_index().copy()
    # 
    #     index = remove_redundant_entries(index)
    # 
    #     # extract info for all entries
    #     index = extract_info(index)
    # 
    #     # save index as a CSV in /path/raw/
    #     raw_path = os.path.join(args.path, 'raw')
    #     os.makedirs(raw_path, exist_ok=True)
    #     index.to_csv(os.path.join(raw_path, 'all_positives.csv'), index=False)
    
    # create fasta files
    # if args.create_fasta:
    #     helper_create_fasta_file(index, args.path)

    # if requested, deleak the dataset by uniprot ids, use already saved .csv-file if possible
    if args.deleak_uniprot:
        print("######### Deleaking dataset by Uniprot IDs #########\n")

        # Load save point with deleaking if exists
        if os.path.exists(os.path.join(args.path, 'deleak_uniprot', 'raw', 'all_positives.csv')):
            index = pd.read_csv(os.path.join(args.path, 'deleak_uniprot', 'raw', 'all_positives.csv'))
            print("Using existing uniprot deleaked Pinder dataframe from CSV. \n")
        else:
            print("Deleaking dataset... \n")
            index = deleak_by_uniprot(index)

            # save deleaked index as a CSV in /path/deleak_uniprot/
            deleak_path = os.path.join(args.path, 'deleak_uniprot', 'raw')
            os.makedirs(deleak_path, exist_ok=True)
            index.to_csv(os.path.join(deleak_path, 'all_positives.csv'), index=False)

    # get different splits
    train = index[index["split"] == "train"]
    val = index[index["split"] == "val"]
    test = index[index["split"] == "test"]

    # deleak with result of cd-hit-2d
    if args.deleak_cdhit:
        print("######### Deleaking dataset by CD-HIT #########\n")
        train, test = deleak_by_cdhit(train, test, args.path)

    # limit train to 10k samples, if specified
    if args.train_lim:
        print(f"Limiting positive training set to {args.train_lim} samples.\n")
        train = train[:args.train_lim]

    # Sample negatives for each split
    neg_train = sample_negatives(train, "train", len(train))
    neg_val   = sample_negatives(val, "val", len(val))
    neg_test  = sample_negatives(test, "test", len(test))

    # Concatenate positive and negative samples
    train_dataset = pd.concat([train, neg_train], ignore_index=True)
    val_dataset   = pd.concat([val, neg_val], ignore_index=True)
    test_dataset  = pd.concat([test, neg_test], ignore_index=True)

    # Save splits
    print("######### Save dataset splits #########\n")
    if not args.deleak_uniprot and not args.deleak_cdhit:
        if not args.train_lim:
            print(f"Save dataset splits to {args.path}")
            train_dataset.to_csv(os.path.join(args.path, 'train.csv'), index=False)
            val_dataset.to_csv(os.path.join(args.path, 'val.csv'), index=False)
            test_dataset.to_csv(os.path.join(args.path, 'test.csv'), index=False)
        else:
            print(f"Save limited dataset splits to {os.path.join(args.path, f'train_{args.train_lim}')}")
            os.makedirs(os.path.join(args.path, f'train_pos_lim_{args.train_lim}'), exist_ok=True)
            train_dataset.to_csv(os.path.join(args.path, f'train_pos_lim_{args.train_lim}', 'train.csv'), index=False)
            val_dataset.to_csv(os.path.join(args.path, f'train_pos_lim_{args.train_lim}', 'val.csv'), index=False)
            test_dataset.to_csv(os.path.join(args.path, f'train_pos_lim_{args.train_lim}', 'test.csv'), index=False)

    elif args.deleak_uniprot and not args.deleak_cdhit:
        if not args.train_lim:
            print(f"Save dataset splits to {os.path.join(args.path, 'deleak_uniprot')}")
            os.makedirs(os.path.join(args.path, 'deleak_uniprot'), exist_ok=True)
            train_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', 'train.csv'), index=False)
            val_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', 'val.csv'), index=False)
            test_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', 'test.csv'), index=False)
        else:
            print(f"Save limited dataset splits to {os.path.join(args.path, f'deleak_uniprot', f'train_{args.train_lim}')}")
            os.makedirs(os.path.join(args.path, 'deleak_uniprot', f'train_pos_lim_{args.train_lim}'), exist_ok=True)
            train_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', f'train_pos_lim_{args.train_lim}', 'train.csv'), index=False)
            val_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', f'train_pos_lim_{args.train_lim}', 'val.csv'), index=False)
            test_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', f'train_pos_lim_{args.train_lim}', 'test.csv'), index=False)

    elif args.deleak_uniprot and args.deleak_cdhit:
        if not args.train_lim:
            print(f"Save dataset splits to {os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit')}")
            os.makedirs(os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit'), exist_ok=True)
            train_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit', 'train.csv'), index=False)
            val_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit', 'val.csv'), index=False)
            test_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit', 'test.csv'), index=False)
        else:
            print(f"Save limited dataset splits to {os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit', f'train_{args.train_lim}')}")
            os.makedirs(os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit', f'train_pos_lim_{args.train_lim}'), exist_ok=True)
            train_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit', f'train_pos_lim_{args.train_lim}', 'train.csv'), index=False)
            val_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit', f'train_pos_lim_{args.train_lim}', 'val.csv'), index=False)
            test_dataset.to_csv(os.path.join(args.path, 'deleak_uniprot', 'deleak_cdhit', f'train_pos_lim_{args.train_lim}', 'test.csv'), index=False)

    elif not args.deleak_uniprot and args.deleak_cdhit:
        if not args.train_lim:
            print(f"Save dataset splits to {os.path.join(args.path, 'deleak_cdhit')}")
            os.makedirs(os.path.join(args.path, 'deleak_cdhit'), exist_ok=True)
            train_dataset.to_csv(os.path.join(args.path, 'deleak_cdhit', 'train.csv'), index=False)
            val_dataset.to_csv(os.path.join(args.path, 'deleak_cdhit', 'val.csv'), index=False)
            test_dataset.to_csv(os.path.join(args.path, 'deleak_cdhit', 'test.csv'), index=False)
        else:
            print(f"Save dataset splits to {os.path.join(args.path, 'deleak_cdhit')}")
            os.makedirs(os.path.join(args.path, 'deleak_cdhit', f'train_pos_lim_{args.train_lim}'), exist_ok=True)
            train_dataset.to_csv(os.path.join(args.path, 'deleak_cdhit', f'train_pos_lim_{args.train_lim}', 'train.csv'), index=False)
            val_dataset.to_csv(os.path.join(args.path, 'deleak_cdhit', f'train_pos_lim_{args.train_lim}', 'val.csv'), index=False)
            test_dataset.to_csv(os.path.join(args.path, 'deleak_cdhit', f'train_pos_lim_{args.train_lim}', 'test.csv'), index=False)










