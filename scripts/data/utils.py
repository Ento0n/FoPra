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
        while len(neg_records) < min(len(non_self_interacting_uni_ids), len(self_interacting_uni_ids), n_samples):
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

    # sample not self interacting until n_samples is reached
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

    print(f"Creating completely balanced splits using splits from: {path}\n")

    # consider only positive samples for deleaking
    train = train[train["label"] == 1]
    val = val[val["label"] == 1]
    test = test[test["label"] == 1]

    # train, val and test should have as many self interacting in positive as in negative -> remove from positive set
    train_self = train[train["receptor_seq"] == train["ligand_seq"]]
    train_nonself = train[train["receptor_seq"] != train["ligand_seq"]]
    train_self_seqs_uniq = set(train_self["receptor_seq"].unique())
    train_nonself_seqs_uniq = set(list(train_nonself["receptor_seq"].unique()) + list(train_nonself["ligand_seq"].unique()))
    print(f"Train has {len(train_self_seqs_uniq)} self interacting sequences and {len(train_nonself_seqs_uniq)} non-self interacting sequences.")
    if len(train_self_seqs_uniq) > len(train_nonself_seqs_uniq):
        #sample len(nonself_seqs_uniq) self interacting sequences
        sampled_seqs = random.sample(list(train_self_seqs_uniq), k=len(train_nonself_seqs_uniq))
        train_self = train_self[train_self["receptor_seq"].isin(sampled_seqs)]
        train = pd.concat([train_self, train_nonself], ignore_index=True)
        print(f"Train should have {len(train_nonself_seqs_uniq)} self interacting sequences to be balanced.")
        print(f"Now train has {len(train[train['receptor_seq'] == train['ligand_seq']]['receptor_seq'].unique())} self interacting sequences.\n")

    # repeat for val set
    val_self = val[val["receptor_seq"] == val["ligand_seq"]]
    val_nonself = val[val["receptor_seq"] != val["ligand_seq"]]
    val_self_seqs_uniq = set(val_self["receptor_seq"].unique())
    val_nonself_seqs_uniq = set(list(val_nonself["receptor_seq"].unique()) + list(val_nonself["ligand_seq"].unique()))

    print(f"Val has {len(val_self_seqs_uniq)} self interacting sequences and {len(val_nonself_seqs_uniq)} non-self interacting sequences.")
    if len(val_self_seqs_uniq) > len(val_nonself_seqs_uniq):
        #sample len(nonself_seqs_uniq) self interacting sequences
        sampled_seqs = random.sample(list(val_self_seqs_uniq), k=len(val_nonself_seqs_uniq))
        val_self = val_self[val_self["receptor_seq"].isin(sampled_seqs)]
        val = pd.concat([val_self, val_nonself], ignore_index=True)
        print(f"Val should have {len(val_nonself_seqs_uniq)} self interacting sequences to be balanced.")
        print(f"Now val has {len(val[val['receptor_seq'] == val['ligand_seq']]['receptor_seq'].unique())} self interacting sequences.\n")


    # repeat for test set
    test_self = test[test["receptor_seq"] == test["ligand_seq"]]
    test_nonself = test[test["receptor_seq"] != test["ligand_seq"]]
    test_self_seqs_uniq = set(test_self["receptor_seq"].unique())
    test_nonself_seqs_uniq = set(list(test_nonself["receptor_seq"].unique()) + list(test_nonself["ligand_seq"].unique()))
    

    print(f"Test has {len(test_self_seqs_uniq)} self interacting sequences and {len(test_nonself_seqs_uniq)} non-self interacting sequences.")
    if len(test_self_seqs_uniq) > len(test_nonself_seqs_uniq):
        #sample len(nonself_seqs_uniq) self interacting sequences
        sampled_seqs = random.sample(list(test_self_seqs_uniq), k=len(test_nonself_seqs_uniq))
        test_self = test_self[test_self["receptor_seq"].isin(sampled_seqs)]
        test = pd.concat([test_self, test_nonself], ignore_index=True)
        print(f"Test should have {len(test_nonself_seqs_uniq)} self interacting sequences to be balanced.")
        print(f"Now test has {len(test[test['receptor_seq'] == test['ligand_seq']]['receptor_seq'].unique())} self interacting sequences.\n")

    # new line for nicer stdout
    print("\n")

    # sample negatives for val and test
    neg_val = sample_negatives(val, split="val", n_samples=len(val), self_interactions=True)
    neg_test = sample_negatives(test, split="test", n_samples=len(test), self_interactions=True)
    neg_train = sample_negatives(train, split="train", n_samples=len(train), self_interactions=True)

    val = pd.concat([val, neg_val], ignore_index=True)
    test = pd.concat([test, neg_test], ignore_index=True)
    train = pd.concat([train, neg_train], ignore_index=True)

    # add test classes
    test = add_test_classes(df=test)

    # save splits
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
    test_df["class"] = test_df.apply(lambda row: "self" if row["receptor_seq"] == row["ligand_seq"] else "non-self", axis=1)
    test_df.insert(3, "class", test_df.pop("class")) # move to 4th column

    # Hnadle special cases based on uniprot ids
    for idx, row in test_df.iterrows():
        rec_uniprot = row["entry"].split("--")[0].split("_")[-1]
        lig_uniprot = row["entry"].split("--")[1].split("_")[-1]

        # edit class when self-interaction with equal uniprot ids
        if rec_uniprot == lig_uniprot and row["class"] != "self":
            test_df.at[idx, "class"] = "uniprot-self"

        # edit class when undefined in uniprot ids
        if rec_uniprot.lower() == "undefined" or lig_uniprot.lower() == "undefined":
            test_df.at[idx, "class"] = "undefined"


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
    else:
        print("Invalid function selected.")