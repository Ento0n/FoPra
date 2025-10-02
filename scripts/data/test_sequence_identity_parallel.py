import os
import sys
import pandas as pd
import edlib
import argparse
from multiprocessing import Pool, cpu_count
from itertools import product


# calculate sequence identity global alignment
def identity_edlib(seq1, seq2):
    """
    Global alignment identity calculation
      - Uses edlib for alignment and calculates identity based on edit distance
    """
    result = edlib.align(seq1, seq2, task="distance")
    distance = result["editDistance"]

    aln_len = max(len(seq1), len(seq2))
    identity = 1 - distance / aln_len

    return identity

# calculate sequence identity with local alignment
def cd_hit_identity_edlib(seq1: str, seq2: str):
    """
    CD-HIT–style identity via edlib:
      - mode="HW": alignment must cover the whole query (the shorter sequence),
        but can start/end anywhere on the target (the longer sequence).
      - Count identical aligned columns (excluding gaps) and divide by len(shorter).
      - Return (identity, distance).
    """

    # Ensure query is the shorter sequence
    if len(seq1) <= len(seq2):
        query, target = seq1, seq2
    else:
        query, target = seq2, seq1

    # Align: path required so we can reconstruct matches
    res = edlib.align(query, target, mode="HW", task="path")

    # Reconstruct aligned strings
    nice = edlib.getNiceAlignment(res, query, target)
    q_aln = nice["query_aligned"]
    t_aln = nice["target_aligned"]

    # Count exact matches in aligned columns
    matches = 0
    for qc, tc in zip(q_aln, t_aln):
        if qc != "-" and tc != "-" and qc == tc:
            matches += 1

    identity = matches / len(query)

    return identity

# parallel max identity
def max_identity_pairwise(set1, set2, n_processes=None):
    """Compute the maximum identity between all seqs in set1 × set2."""
    if n_processes is None:
        n_processes = cpu_count()

    print(f"Using {n_processes} processes for parallel computation.\n")

    # Create all pairwise combinations
    pairwise_combinations = list(product(set1, set2))

    with Pool(n_processes) as pool:
        identities = pool.starmap(cd_hit_identity_edlib, pairwise_combinations)

    avg = sum(identities) / len(identities)
    max_val = max(identities)
    max_idx = identities.index(max_val)
    max_pair = pairwise_combinations[max_idx]

    return max_val, avg, max_pair

def main():
    print("######### Test sequence identity between dataset splits #########\n")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test sequence identity between dataset splits")
    parser.add_argument('--deleak_cdhit', action='store_true', help='Use by cdhit deleaked dataset')
    parser.add_argument('--deleak_uniprot', action='store_true', help='Use by uniprot deleaked dataset')
    parser.add_argument('--path', type=str, default='/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/', help='Base path for datasets')
    parser.add_argument('--judith_test', action='store_true', help='Use the judith test set created by create_judith_test.py')

    # Add mutually exclusive arguments for fasta creation and deleaking
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--global', action='store_true', help='Use global alignemnt for identity')
    group.add_argument('--local', action='store_true', help='Use local alignment for identity')

    args = parser.parse_args()

    # figure out which CSV files to use
    if not args.deleak_uniprot and not args.deleak_cdhit:
        train_csv = os.path.join(args.path, 'train.csv')
        val_csv = os.path.join(args.path, 'val.csv')
        test_csv = os.path.join(args.path, 'test.csv')
    elif args.deleak_uniprot and not args.deleak_cdhit:
        train_csv = os.path.join(args.path, 'deleak_uniprot', 'train.csv')
        val_csv = os.path.join(args.path, 'deleak_uniprot', 'val.csv')
        test_csv = os.path.join(args.path, 'deleak_uniprot', 'test.csv')
    elif not args.deleak_uniprot and args.deleak_cdhit:
        train_csv = os.path.join(args.path, f'deleak_cdhit', 'train.csv')
        val_csv = os.path.join(args.path, f'deleak_cdhit', 'val.csv')
        test_csv = os.path.join(args.path, f'deleak_cdhit', 'test.csv')
    elif args.deleak_uniprot and args.deleak_cdhit:
        train_csv = os.path.join(args.path, f'deleak_uniprot', 'deleak_cdhit', 'train.csv')
        val_csv = os.path.join(args.path, f'deleak_uniprot', 'deleak_cdhit', 'val.csv')
        test_csv = os.path.join(args.path, f'deleak_uniprot', 'deleak_cdhit', 'test.csv')
    else:
        print("This should never be reached. Check source code! :O")
        sys.exit()

    if args.judith_test:
        test_csv = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/test_pinder.csv"

    # extract needed columns
    cols = ["receptor_seq", "ligand_seq"]
    train_df = pd.read_csv(train_csv, usecols=cols)
    val_df = pd.read_csv(val_csv, usecols=cols)
    test_df = pd.read_csv(test_csv, usecols=cols)

    # Ensure sequences are unique across datasets
    train_seqs = pd.concat([train_df['receptor_seq'], train_df['ligand_seq']]).drop_duplicates().tolist()
    val_seqs = pd.concat([val_df['receptor_seq'], val_df['ligand_seq']]).drop_duplicates().tolist()
    test_seqs = pd.concat([test_df['receptor_seq'], test_df['ligand_seq']]).drop_duplicates().tolist()

    # Compute maximum sequence identity
    max_train_val, avg_train_val, max_train_val_pair = max_identity_pairwise(train_seqs, val_seqs)
    max_train_test, avg_train_test, max_train_test_pair = max_identity_pairwise(train_seqs, test_seqs)
    max_val_test, avg_val_test, max_val_test_pair = max_identity_pairwise(val_seqs, test_seqs)

    # print collected results
    print(f"Using dataset splits from:\nTrain: {train_csv}\nVal: {val_csv}\nTest: {test_csv}\n")
    print("Results:")
    print(f"Max sequence identity between train/val: {max_train_val:.4f}")
    print(f"Max sequence identity pair between train/val: {max_train_val_pair}")
    print(f"Avg sequence identity between train/val: {avg_train_val:.4f}\n")
    print(f"Max sequence identity between train/test: {max_train_test:.4f}")
    print(f"Max sequence identity pair between train/test: {max_train_test_pair}")
    print(f"Avg sequence identity between train/test: {avg_train_test:.4f}\n")
    print(f"Max sequence identity between val/test: {max_val_test:.4f}")
    print(f"Max sequence identity pair between val/test: {max_val_test_pair}")
    print(f"Avg sequence identity between val/test: {avg_val_test:.4f}\n")

if __name__ == "__main__":
    main()