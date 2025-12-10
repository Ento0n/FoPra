import sys
import pandas as pd
import os
import argparse
from collections import Counter
import random

# self interactions doesn't consider duplicate interactions
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
    # Implemented for unique self interactions only (A-A once, B-B once, no duplicates)
    neg_records = []
    neg_set = set()
    target_self_interactions = 0
    i = 0
    if self_interactions:
        # extract the different sequences associated to a uniprot ID not in positives self interactions
        uni_id_to_seqs = {}
        self_interacting_uni_ids = set()
        blacklist_uni_seqs = set()
        for _, row in df.iterrows():
            uni_rec = row['entry'].split('--')[0].split('_')[-1]
            uni_lig = row['entry'].split('--')[1].split('_')[-1]

            if uni_rec == uni_lig:
                # Add to self interacting uniprot IDs and blacklist
                self_interacting_uni_ids.add(uni_rec)
                blacklist_uni_seqs.add(row["receptor_seq"])
                blacklist_uni_seqs.add(row["ligand_seq"])

                # remove sequences from uni_id_to_seqs if already present, only receptor needed since same as ligand here. But both sequences matter!!
                if uni_rec in uni_id_to_seqs:
                    if row["receptor_seq"] in uni_id_to_seqs[uni_rec]:
                        uni_id_to_seqs[uni_rec].remove(row["receptor_seq"])
                    if row["ligand_seq"] in uni_id_to_seqs[uni_rec]:
                        uni_id_to_seqs[uni_rec].remove(row["ligand_seq"])

                # skip rest of loop
                continue
            
            # Add sequences to uniprot ID mapping
            if row["receptor_seq"] not in blacklist_uni_seqs:
                if uni_rec not in uni_id_to_seqs and row['receptor_seq'] not in blacklist_uni_seqs:
                    uni_id_to_seqs[uni_rec] = set()
                uni_id_to_seqs[uni_rec].add(row['receptor_seq'])

            if row["ligand_seq"] not in blacklist_uni_seqs:
                if uni_lig not in uni_id_to_seqs and row['ligand_seq'] not in blacklist_uni_seqs:
                    uni_id_to_seqs[uni_lig] = set()
                uni_id_to_seqs[uni_lig].add(row['ligand_seq'])
        
        # uniprot IDs not self interacting
        non_self_interacting_uni_ids = set(uni_id_to_seqs.keys()) - self_interacting_uni_ids

        print(f"# of unique uniprot IDs in positive self interactions: {len(self_interacting_uni_ids)}")
        print(f"# of possible non-self interacting uniprot IDs to add as negatives in split {split}: {len(non_self_interacting_uni_ids)}")
        print(f"Sanity check - intersection: {self_interacting_uni_ids.intersection(non_self_interacting_uni_ids)}")

        # go through non self interacting uniprot IDs and add their sequences as self interactions
        target_self_interactions = min(len(non_self_interacting_uni_ids), len(self_interacting_uni_ids), n_samples)
        print(f"Sampling {target_self_interactions} self-interacting negative samples for split {split}...")
        while len(neg_records) < target_self_interactions:
            uni_id = non_self_interacting_uni_ids.pop()
            seqs = list(uni_id_to_seqs[uni_id])
            seq1 = random.choice(seqs)
            seq2 = random.choice(seqs)

            # create record with paths if available
            if path:
                rec_path_obj = df.loc[df['receptor_seq']==seq1, 'receptor_path']
                lig_path_obj = df.loc[df['ligand_seq']==seq2,   'ligand_path']
                
                # seq1 and seq2 can be set that above leads to 2 empty path objs, need to check both
                if rec_path_obj.empty and lig_path_obj.empty:
                    rec_path_obj = df.loc[df['receptor_seq']==seq2, 'receptor_path']
                    lig_path_obj = df.loc[df['ligand_seq']==seq1,   'ligand_path']

                if not rec_path_obj.empty:
                    rec_path = rec_path_obj.iat[0]
                    lig_path = rec_path_obj.iat[0]
                elif not lig_path_obj.empty:
                    rec_path = lig_path_obj.iat[0]
                    lig_path = lig_path_obj.iat[0]
                else:
                    print(f"Warning: Could not find path for self-interacting sequence {seq1} or {seq2} in split {split}. Paths will be set to None.")
                    rec_path = None
                    lig_path = None
                
                i += 1
                neg_records.append({
                    'entry': "XXXX__XX_" + uni_id + '--' + "XXXX__XX_" + uni_id,
                    'split': split,
                    'receptor_seq': seq1,
                    'ligand_seq':  seq2,
                    'receptor_path': rec_path,
                    'ligand_path':  lig_path,
                    'label': 0
                })
            else:
                i += 1
                neg_records.append({
                    'entry': "XXXX__XX_" + uni_id + '--' + "XXXX__XX_" + uni_id,
                    'split': split,
                    'receptor_seq': seq1,
                    'ligand_seq':  seq2,
                    'label': 0
                })
            
            neg_set.add((seq1, seq2))

            # reduce weights accordingly
            if seq1 in rec_seqs:
                rec_index = rec_seqs.index(seq1)
                rec_weights[rec_index] = max(0, rec_weights[rec_index] - 1)
            
            if seq2 in lig_seqs:
                lig_index = lig_seqs.index(seq2)
                lig_weights[lig_index] = max(0, lig_weights[lig_index] - 1)
    
    print(f"Counter i is {i}, expected {target_self_interactions}")

    # sample not self interacting until n_samples is reached
    print(f"Sampling remaining {n_samples - len(neg_records)} negative samples for split {split}...\n")
    while len(neg_records) < n_samples:
        rec_seq = random.choices(rec_seqs, weights=rec_weights, k=1)[0]
        lig_seq = random.choices(lig_seqs, weights=lig_weights, k=1)[0]
        
        if (rec_seq, lig_seq) in positives or (lig_seq, rec_seq) in positives:
            continue
            
        if (rec_seq, lig_seq) in neg_set or (lig_seq, rec_seq) in neg_set:
            continue

        if self_interactions:
            # already added self interactions
            if rec_seq == lig_seq:
                continue
        
        rec_entry = df.loc[df['receptor_seq']==rec_seq, 'entry'].iat[0].split("--")[0]
        lig_entry = df.loc[df['ligand_seq']==lig_seq, 'entry'].iat[0].split("--")[1]
        entry = rec_entry + '--' + lig_entry

        if path:
            rec_path = df.loc[df['receptor_seq']==rec_seq, 'receptor_path'].iat[0]
            lig_path = df.loc[df['ligand_seq']==lig_seq,   'ligand_path'].iat[0]
            neg_records.append({
                'entry': entry,
                'split': split,
                'receptor_seq': rec_seq,
                'ligand_seq':  lig_seq,
                'receptor_path': rec_path,
                'ligand_path':  lig_path,
                'label': 0
            })
            neg_set.add((rec_seq, lig_seq))
        else:
            neg_records.append({
                'entry': entry,
                'split': split,
                'receptor_seq': rec_seq,
                'ligand_seq':  lig_seq,
                'label': 0
            })
            neg_set.add((rec_seq, lig_seq))

    return pd.DataFrame(neg_records)

def resample_negatives(path: str, half_balance: bool) -> None:
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val   = pd.read_csv(os.path.join(path, "val.csv"))
    test  = pd.read_csv(os.path.join(path, "test.csv"))

    train_pos = train[train["label"] == 1]
    val_pos   = val[val["label"] == 1]
    test_pos  = test[test["label"] == 1]

    if not half_balance:
        train_neg = sample_negatives(train_pos, "train", len(train_pos))
        val_neg   = sample_negatives(val_pos,   "val",   len(val_pos))
        test_neg  = sample_negatives(test_pos,  "test",  len(test_pos))
    else:
        train_neg = sample_negatives(train_pos, "train", len(train_pos), self_interactions=True)
        val_neg   = sample_negatives(val_pos,   "val",   len(val_pos),   self_interactions=True)
        test_neg  = sample_negatives(test_pos,  "test",  len(test_pos),  self_interactions=True)

    train = pd.concat([train_pos, train_neg], ignore_index=True)
    val   = pd.concat([val_pos,   val_neg],   ignore_index=True)
    test  = pd.concat([test_pos,  test_neg],  ignore_index=True)

    # add test classes
    test = add_test_classes(df=test)

    train.to_csv(os.path.join(path, "train.csv"), index=False)
    val.to_csv(os.path.join(path, "val.csv"), index=False)
    test.to_csv(os.path.join(path, "test.csv"), index=False)

def completely_balanced_splits(path: str, out_path: str):
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val = pd.read_csv(os.path.join(path, "val.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))

    print(f"Original splits from: {path}\n")

    # Only consider positive samples for deleaking
    train = train[train["label"] == 1]
    val = val[val["label"] == 1]
    test = test[test["label"] == 1]

    def remove_excess_self_interactions(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        uni_rec = df["entry"].str.split('--').str[0].str.split('_').str[-1]
        uni_lig = df["entry"].str.split('--').str[1].str.split('_').str[-1]

        self_interacting = df[uni_rec == uni_lig]

        uni_ids_self = set(uni_rec[uni_rec == uni_lig].unique())
        uni_ids_nonself = set(uni_rec).union(set(uni_lig)) - uni_ids_self

        if len(uni_ids_self) > len(uni_ids_nonself):
            print(f"{split_name} has {len(uni_ids_self)} self interacting uniprot IDs and {len(uni_ids_nonself)} non-self interacting uniprot IDs.")
            print(f"Sampling {len(uni_ids_nonself)} self interacting uniprot IDs to balance.")
            sampled_uni_ids = random.sample(list(uni_ids_self), k=len(uni_ids_nonself))
            self_interacting = self_interacting[uni_rec.isin(sampled_uni_ids)]
            df = pd.concat([self_interacting, df[uni_rec != uni_lig]], ignore_index=True)
            print(f"Now {split_name} has {len(uni_ids_nonself)} self interacting uniprot IDs in positive.\n")
        
        return df

    train = remove_excess_self_interactions(train, "Train")
    val = remove_excess_self_interactions(val, "Val")
    test = remove_excess_self_interactions(test, "Test")

    # resample negatives
    train_neg = sample_negatives(train, "train", len(train), self_interactions=True)
    val_neg = sample_negatives(val, "val", len(val), self_interactions=True)
    test_neg = sample_negatives(test, "test", len(test), self_interactions=True)

    train = pd.concat([train, train_neg], ignore_index=True)
    val = pd.concat([val, val_neg], ignore_index=True)
    test = pd.concat([test, test_neg], ignore_index=True)

    # add test classes right away
    test = add_test_classes(df=test)

    train.to_csv(os.path.join(out_path, "train.csv"), index=False)
    val.to_csv(os.path.join(out_path, "val.csv"), index=False)
    test.to_csv(os.path.join(out_path, "test.csv"), index=False)

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
    elif path is not None:
        test_df = pd.read_csv(os.path.join(path, "test.csv"))
    elif df is not None:
        test_df = df
    else:
        raise ValueError("Unexpected error in input parameters.")

    # add self-interaction class
    class_records = []
    for _, row in test_df.iterrows():
        uni_rec = row["entry"].split("--")[0].split("_")[-1]
        uni_lig = row["entry"].split("--")[1].split("_")[-1]

        if uni_rec == uni_lig:
            class_records.append("self")
        else:
            class_records.append("non-self")
        
        # overwrite case where UNDEFINED
        if uni_rec.lower() == "undefined" or uni_lig.lower() == "undefined":
            class_records[-1] = "undefined"

    test_df["class"] = class_records

    # reorder columns to have class after label
    test_df.insert(3, "class", test_df.pop("class")) # move to 4th column

    # save updated test_df
    if path is not None:
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

    # bidirectional dataset requires removing both A-B and B-A interactions
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

def add_split_column(path: str) -> pd.DataFrame:
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val   = pd.read_csv(os.path.join(path, "val.csv"))
    test  = pd.read_csv(os.path.join(path, "test.csv"))

    train["split"] = "train"
    val["split"]   = "val"
    test["split"]  = "test"

    train.to_csv(os.path.join(path, "train.csv"), index=False)
    val.to_csv(os.path.join(path, "val.csv"), index=False)
    test.to_csv(os.path.join(path, "test.csv"), index=False)

def order_columns(path: str) -> None:
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val   = pd.read_csv(os.path.join(path, "val.csv"))
    test  = pd.read_csv(os.path.join(path, "test.csv"))

    column_order = ["entry", "split", "label", "receptor_seq", "ligand_seq", "receptor_path", "ligand_path"]
    column_order_test = ["entry", "split", "label", "class", "receptor_seq", "ligand_seq", "receptor_path", "ligand_path"]

    train = train[column_order]
    val   = val[column_order]
    test  = test[column_order_test]

    train.to_csv(os.path.join(path, "train.csv"), index=False)
    val.to_csv(os.path.join(path, "val.csv"), index=False)
    test.to_csv(os.path.join(path, "test.csv"), index=False)

def remove_too_short_seqs(path: str) -> None:
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val   = pd.read_csv(os.path.join(path, "val.csv"))
    test  = pd.read_csv(os.path.join(path, "test.csv"))

    train = train[train["label"] == 1]
    val = val[val["label"] == 1]
    test = test[test["label"] == 1]

    # remove everything smaller than 5 amino acids
    train = train[(train["receptor_seq"].str.len() > 5) & (train["ligand_seq"].str.len() > 5)]
    val = val[(val["receptor_seq"].str.len() > 5) & (val["ligand_seq"].str.len() > 5)]
    test = test[(test["receptor_seq"].str.len() > 5) & (test["ligand_seq"].str.len() > 5)]

    # sample negatives again
    train_neg = sample_negatives(train, "train", len(train))
    val_neg = sample_negatives(val, "val", len(val))
    test_neg = sample_negatives(test, "test", len(test))

    train = pd.concat([train, train_neg], ignore_index=True)
    val = pd.concat([val, val_neg], ignore_index=True)
    test = pd.concat([test, test_neg], ignore_index=True)

    # add test classes right away
    test = add_test_classes(df=test)

    train.to_csv(os.path.join(path, "train.csv"), index=False)
    val.to_csv(os.path.join(path, "val.csv"), index=False)
    test.to_csv(os.path.join(path, "test.csv"), index=False)

def extract_train_val_seqs(path: str, out_path: str) -> None:
    df = pd.concat([
        pd.read_csv(os.path.join(path, "train.csv")),
        pd.read_csv(os.path.join(path, "val.csv")),
    ])

    sequences = pd.concat([df["receptor_seq"], df["ligand_seq"]]).unique()
    with open(os.path.join(out_path, "train_val_sequences.fasta"), "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")

def remove_overlapping_uniprot_ids(path: str, out_path: str) -> None:
    train_csv = os.path.join(path, "train.csv")
    val_csv = os.path.join(path, "val.csv")
    test_csv = os.path.join(path, "test.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # only positives considered for UNIPROT ID overlaps, but only train and test cleaned
    train_df = train_df[train_df["label"] == 1]
    test_df = test_df[test_df["label"] == 1]

    # extract unique UNIPROT IDs
    def extract_unique_uniprot_ids(df):
        df["receptor_uniprot_id"] = df["entry"].str.split("--").str[0].str.split("_").str[-1]
        df["ligand_uniprot_id"] = df["entry"].str.split("--").str[1].str.split("_").str[-1]

        receptor_ids = set(df["receptor_uniprot_id"].unique())
        ligand_ids = set(df["ligand_uniprot_id"].unique())

        unique_ids = receptor_ids.union(ligand_ids)
        return unique_ids

    train_ids = extract_unique_uniprot_ids(train_df)
    val_ids = extract_unique_uniprot_ids(val_df)
    test_ids = extract_unique_uniprot_ids(test_df)

    print(f"Number of unique UNIPROT IDs in Train: {len(train_ids)}")
    print(f"Number of unique UNIPROT IDs in Val: {len(val_ids)}")
    print(f"Number of unique UNIPROT IDs in Test: {len(test_ids)}")

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    train_overlap = train_val_overlap.union(train_test_overlap)
    val_test_overlap = val_ids.intersection(test_ids)

    # print overlaps between sets
    print(f"Train Val & Test overlap ({len(train_overlap)}): {train_overlap}")
    print(f"Val and Test overlap ({len(val_test_overlap)}): {val_test_overlap}")

    # remove overlapping UNIPROT IDs from train set
    def remove_overlapping_ids(df, overlapping_ids):
        mask_rec = df["entry"].str.split("--").str[0].str.split("_").str[-1].isin(overlapping_ids)
        mask_lig = df["entry"].str.split("--").str[1].str.split("_").str[-1].isin(overlapping_ids)
        combined_mask = mask_rec | mask_lig
        filtered_df = df[~combined_mask].reset_index(drop=True)
        return filtered_df
    
    cleaned_train_df = remove_overlapping_ids(train_df, train_overlap)
    print(f"Size of Train set before removing overlapping UNIPROT IDs: {len(train_df)}")
    print(f"Size of Train set after removing overlapping UNIPROT IDs: {len(cleaned_train_df)}")
    
    cleaned_test_df = remove_overlapping_ids(test_df, val_test_overlap)
    print(f"Size of Test set before removing overlapping UNIPROT IDs: {len(test_df)}")
    print(f"Size of Test set after removing overlapping UNIPROT IDs: {len(cleaned_test_df)}")

    print(f"Val set remains unchanged with size: {len(val_df)}")

    # resample negatives
    cleaned_train_neg = sample_negatives(cleaned_train_df, "train", len(cleaned_train_df))
    cleaned_train_df = pd.concat([cleaned_train_df, cleaned_train_neg], ignore_index=True)
    cleaned_test_neg = sample_negatives(cleaned_test_df, "test", len(cleaned_test_df))
    cleaned_test_df = pd.concat([cleaned_test_df, cleaned_test_neg], ignore_index=True)

    # save cleaned datasets
    cleaned_train_df.to_csv(os.path.join(out_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_path, "val.csv"), index=False)
    cleaned_test_df.to_csv(os.path.join(out_path, "test.csv"), index=False)

def extract_test_species():
    test_file_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/no_duplicates/deleak_cdhit/fully_balanced/test.csv"
    uniprot_accession_tax_id_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/no_duplicates/deleak_cdhit/sunburst_data/full_uniprot_sequences.fasta"
    lineage_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/no_duplicates/deleak_cdhit/sunburst_data/uniprot_lineage.tsv"
    out_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/no_duplicates/deleak_cdhit/fully_balanced/test_species_counts.tsv"

    accession_tax_id_mapping = {}
    with open(uniprot_accession_tax_id_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                accession = line.split("|")[0][1:].strip()
                tax_id = line.split("|")[1].split("=")[1].strip()
                accession_tax_id_mapping[accession] = tax_id
    
    print(f"Total accession to tax ID mappings: {len(accession_tax_id_mapping)}")

    tax_id_scientific_name_mapping = {}
    with open(lineage_path, "r") as f:
        for line in f:
            tax_id = line.split("\t")[0].strip()
            scientific_name = line.split("\t")[1].strip()
            tax_id_scientific_name_mapping[tax_id] = scientific_name

    print(f"Total tax ID to scientific name mappings: {len(tax_id_scientific_name_mapping)}")
    test_df = pd.read_csv(test_file_path)

    # Count for each species in test how often it appears
    tax_id_counter = Counter()
    for _, row in test_df.iterrows():
        rec_accession = row["entry"].split("--")[0].split("_")[-1]
        lig_accession = row["entry"].split("--")[1].split("_")[-1]

        if rec_accession != "UNDEFINED":
            rec_tax_id = accession_tax_id_mapping[rec_accession]
            tax_id_counter[rec_tax_id] += 1

        if lig_accession != "UNDEFINED":
            lig_tax_id = accession_tax_id_mapping[lig_accession]
            tax_id_counter[lig_tax_id] += 1
    
    # print species counts to out_path
    with open(out_path, "w") as f:
        f.write("Tax ID\tScientific Name\tCount in Test Set\n")
        for tax_id, count in tax_id_counter.most_common():
            if tax_id in tax_id_scientific_name_mapping.keys():
                scientific_name = tax_id_scientific_name_mapping[tax_id]
            else:
                scientific_name = "UNKNOWN"
            f.write(f"{tax_id}\t{scientific_name}\t{count}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="different utilities for data processing")
    parser.add_argument('--function', type=int, required=True, help='Function to execute: 1 - add labels, 2 - add test classes, 3 - create fasta files')
    parser.add_argument('--path', type=str, help='Path to read from')
    parser.add_argument('--out_path', type=str, help='Path to save to')
    parser.add_argument('--half_balance', action='store_true', help='Whether to use half balancing when resampling negatives')
    args = parser.parse_args()

    if args.function == 1:
        add_labels()
    elif args.function == 2:
        add_test_classes(args.path)
    elif args.function == 3:
        helper_create_fasta_file(args.path, args.out_path)
    elif args.function == 4:
        helper_remove_duplicate_interactions(args.path, args.out_path)
    elif args.function == 5:
        resample_negatives(args.path, args.half_balance)
    elif args.function == 6:
        add_split_column(args.path)
    elif args.function == 7:
        order_columns(args.path)
    elif args.function == 8:
        completely_balanced_splits(args.path, args.out_path)
    elif args.function == 9:
        remove_too_short_seqs(args.path)
    elif args.function == 10:
        extract_train_val_seqs(args.path, args.out_path)
    elif args.function == 11:
        remove_overlapping_uniprot_ids(args.path, args.out_path)
    elif args.function == 12:
        extract_test_species()
    else:
        print("Invalid function selected.")
