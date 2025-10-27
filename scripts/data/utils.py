import pandas as pd
import os
import argparse

from create_dataset import sample_negatives

def add_labels():
    identity_df_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/test_with_identities_raw.csv"
    origin_df_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/test.csv"
    identity_df = pd.read_csv(identity_df_path)
    origin_df = pd.read_csv(origin_df_path)

    # add sequence identity to origin_df
    for idx, row in origin_df.iterrows():
        receptor_seq = row["receptor_seq"]
        ligand_seq = row["ligand_seq"]

        mask = (identity_df["receptor_seq"] == receptor_seq) & (identity_df["ligand_seq"] == ligand_seq)
        if mask.any():
            origin_df.at[idx, "max_id_to_train_receptor"] = identity_df.loc[mask, "max_id_to_train_receptor"].values[0]
            origin_df.at[idx, "max_id_to_train_ligand"] = identity_df.loc[mask, "max_id_to_train_ligand"].values[0]
        else:
            print(f"Could not find identity for {row['entry']}")
    
    origin_df.to_csv(os.path.join("/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit", "test_with_identities.csv"), index=False)

def add_test_classes(path: str = None, df: pd.DataFrame = None) -> None:
    if df is None and path is None:
        raise ValueError("Either 'df' or 'path' must be provided.")
    elif path:
        test_df = pd.read_csv(os.path.join(path, "test.csv"))
    elif df:
        test_df = df

    # add self-interaction class
    test_df["class"] = test_df.apply(lambda row: "self" if row["receptor_seq"] == row["ligand_seq"] else "non-self", axis=1)

    # create sequence to uniprot mapping
    seq_to_uniprot = {}
    test_pos = test_df[test_df["label"] == 1]
    for _, row in test_pos.iterrows():
        rec_uniprot = row["entry"].split("--")[0].split("_")[-1]
        lig_uniprot = row["entry"].split("--")[1].split("_")[-1]

        seq_to_uniprot[row["receptor_seq"]] = rec_uniprot
        seq_to_uniprot[row["ligand_seq"]] = lig_uniprot

    # add UNDEFINED class based on uniprot accession
    test_df["class"] = test_df.apply(lambda row: "undefined" if seq_to_uniprot[row["receptor_seq"]] == "UNDEFINED" or seq_to_uniprot[row["ligand_seq"]] == "UNDEFINED" else row["class"], axis=1)

    # save updated test_df
    if path:
        test_df.to_csv(os.path.join(path, "test.csv"), index=False)
        return None
    else:
        return test_df

def helper_create_fasta_file(path: str, out_path: str) -> None:
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

    train = pd.read_csv(os.path.join(path, "train.csv"))
    val   = pd.read_csv(os.path.join(path, "val.csv"))
    test  = pd.read_csv(os.path.join(path, "test.csv"))

    print("Fasta files for the splits train, val & test are created...")
    create_fasta_file(train, 'train', out_path)
    create_fasta_file(val,   'val', out_path)
    create_fasta_file(test,  'test', out_path)
    print("Fasta files created succesfully, finish execution! \n")

def create_fasta_file(df: pd.DataFrame, split: str, out_path: str) -> None:
    """
    Create a FASTA file for a given split, containing all unique receptor and ligand sequences.

    Args:
        df (pd.DataFrame): DataFrame for the split.
        split (str): Name of the split (e.g., 'train', 'val', 'test').
        path (str): Directory to save the FASTA file.

    Returns:
        None
    """

    all_seqs_unique = set(list(df['receptor_seq']) + list(df['ligand_seq']))

    with open(os.path.join(out_path, f"{split}.fasta"), 'w') as f:
        for i, seq in enumerate(all_seqs_unique):
            f.write(f">{f"{i}_{split}"}\n{seq}\n")

def remove_duplicate_interactions(df: pd.DataFrame, split: str) -> pd.DataFrame:
    print(f"Removing duplicate interactions in {split} set, size before: {len(df)}")
    df = df.drop_duplicates(subset=['receptor_seq', 'ligand_seq'])
    df = df.drop_duplicates(subset=['ligand_seq', 'receptor_seq'])
    print(f"Size after: {len(df)}\n")
    return df

def helper_remove_duplicate_interactions(path: str, out_path: str):
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val = pd.read_csv(os.path.join(path, "val.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))

    train = train[train["label"] == 1]
    val = val[val["label"] == 1]
    test = test[test["label"] == 1]

    train = remove_duplicate_interactions(train, "train")
    val = remove_duplicate_interactions(val, "val")
    test = remove_duplicate_interactions(test, "test")

    train_neg = sample_negatives(train, "train", len(train))
    val_neg = sample_negatives(val, "val", len(val))
    test_neg = sample_negatives(test, "test", len(test))

    train = pd.concat([train, train_neg], ignore_index=True)
    val = pd.concat([val, val_neg], ignore_index=True)
    test = pd.concat([test, test_neg], ignore_index=True)

    # add test classes right away
    test = add_test_classes(df=test)

    train.to_csv(os.path.join(out_path, "train.csv"), index=False)
    val.to_csv(os.path.join(out_path, "val.csv"), index=False)
    test.to_csv(os.path.join(out_path, "test.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="different utilities for data processing")
    parser.add_argument('--function', type=int, required=True, help='Function to execute: 1 - add labels, 2 - add test classes, 3 - create fasta files')
    parser.add_argument('--path', type=str, help='Path to read from')
    parser.add_argument('--out_path', type=str, help='Path to save to')
    args = parser.parse_args()

    if args.function == 1:
        add_labels()
    elif args.function == 2:
        add_test_classes(args.path)
    elif args.function == 3:
        helper_create_fasta_file(args.path, args.out_path)
    elif args.function == 4:
        helper_remove_duplicate_interactions(args.path, args.out_path)