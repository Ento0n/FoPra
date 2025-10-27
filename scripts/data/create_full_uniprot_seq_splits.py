import pandas as pd
import os
import random

from create_dataset import sample_negatives, remove_similar_sequences

def replace_sequences(df, seq_dict):
    missing = {}
    for idx, row in df.iterrows():
        rec_uniprot = row["entry"].split("--")[0].split("_")[-1]
        lig_uniprot = row["entry"].split("--")[1].split("_")[-1]

        # handle receptor sequence
        if rec_uniprot in seq_dict.keys():
            df.at[idx, "receptor_seq"] = seq_dict[rec_uniprot]
        else:
            if rec_uniprot not in missing.keys():
                missing[rec_uniprot] = 0
            else:
                missing[rec_uniprot] += 1

        # handle ligand sequence
        if lig_uniprot in seq_dict.keys():
            df.at[idx, "ligand_seq"] = seq_dict[lig_uniprot]
        else:
            if lig_uniprot not in missing.keys():
                missing[lig_uniprot] = 0
            else:
                missing[lig_uniprot] += 1
    
    if len(missing.keys()) > 0:
        print(f"Missing {len(missing.keys())} sequences:")
        for k, v in missing.items():
            print(f"{k}: {v}")
        print("\n")

    return df

def create_full_uniprot_seq_splits():
    splits_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit"
    train_df = pd.read_csv(os.path.join(splits_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_path, "test.csv"))

    train_df = train_df[train_df["label"] == 1]
    val_df = val_df[val_df["label"] == 1]
    test_df = test_df[test_df["label"] == 1]

    uniprot_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/full_uniprot_sequences"

    full_fasta = os.path.join(uniprot_path, "full_uniprot_sequences.fasta")

    # read full fasta into dictionary
    full_seqs = {}
    with open(full_fasta, "r") as f:
        uid = None
        seq = []
        for line in f:
            if line.startswith(">"):
                if uid not in full_seqs.keys() and uid is not None:
                    full_seqs[uid] = "".join(seq)
                    seq = []

                uid = line[1:].strip()
            else:
                seq.append(line.strip())
        
        # add last one
        if uid is not None and uid not in full_seqs.keys():
            print(f"Adding last one: {uid}")
            full_seqs[uid] = "".join(seq)
    
    print(f"full seqs has {len(full_seqs)} entries\n")

    train_df = replace_sequences(train_df, full_seqs)
    val_df = replace_sequences(val_df, full_seqs)
    test_df = replace_sequences(test_df, full_seqs)

    # Remove any rows with sequences > 2500 amino acids
    print("Filtering sequences longer than 2500 amino acids...\n")
    print(f"Before filtering, train has {len(train_df)} entries, val has {len(val_df)} entries, test has {len(test_df)} entries.\n")
    train_df = train_df[(train_df["receptor_seq"].str.len() <= 2500) & (train_df["ligand_seq"].str.len() <= 2500)]
    val_df = val_df[(val_df["receptor_seq"].str.len() <= 2500) & (val_df["ligand_seq"].str.len() <= 2500)]
    test_df = test_df[(test_df["receptor_seq"].str.len() <= 2500) & (test_df["ligand_seq"].str.len() <= 2500)]
    print(f"After filtering, train has {len(train_df)} entries, val has {len(val_df)} entries, test has {len(test_df)} entries.\n")

    train_neg = sample_negatives(train_df, split="train", n_samples=len(train_df))
    val_neg = sample_negatives(val_df, split="val", n_samples=len(val_df))
    test_neg = sample_negatives(test_df, split="test", n_samples=len(test_df))

    train_df = pd.concat([train_df, train_neg], ignore_index=True)
    val_df = pd.concat([val_df, val_neg], ignore_index=True)
    test_df = pd.concat([test_df, test_neg], ignore_index=True)

    train_df.to_csv(os.path.join(uniprot_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(uniprot_path, "val.csv"), index=False)
    test_df.to_csv(os.path.join(uniprot_path, "test.csv"), index=False)

def add_sequences_from_uniparc():
    path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/unique_sequences_uniprot_id"
    tsv = os.path.join(path, "uniprot_uniparc_mapping.txt")

    # read tsv file
    with open(tsv, "r") as f:
        seqs = {}
        for line in f:
            # skip first line
            if line.startswith("From"):
                continue

            # parse line
            parts = line.strip().split("\t")
            uniprot_id = parts[0]
            sequence = parts[2]

            seqs[uniprot_id] = sequence
    
    print(f"Adding {len(seqs)} sequences from uniparc to full_uniprot_sequences.fasta")
    
    # append to full_uniprot_sequences.fasta
    fasta = os.path.join(path, "uniparc_sequences.fasta")
    with open(fasta, "w") as f:
        for uid, seq in seqs.items():
            f.write(f">{uid}\n")
            f.write(f"{seq}\n")

def generate_fasta_splits():
    # Read train, val and test sequences
    path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/unique_sequences_uniprot_id"
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val = pd.read_csv(os.path.join(path, "val.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))

    train_seqs = pd.concat([train['receptor_seq'], train['ligand_seq']]).drop_duplicates().tolist()
    val_seqs = pd.concat([val['receptor_seq'], val['ligand_seq']]).drop_duplicates().tolist()
    test_seqs = pd.concat([test['receptor_seq'], test['ligand_seq']]).drop_duplicates().tolist()

    # write to fasta files
    out_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/unique_sequences_uniprot_id/cd_hit"
    with open(os.path.join(out_path, "train.fasta"), "w") as f:
        for i, seq in enumerate(train_seqs):
            f.write(f">{i}_train\n{seq}\n")

    with open(os.path.join(out_path, "val.fasta"), "w") as f:
        for i, seq in enumerate(val_seqs):
            f.write(f">{i}_val\n{seq}\n")

    with open(os.path.join(out_path, "test.fasta"), "w") as f:
        for i, seq in enumerate(test_seqs):
            f.write(f">{i}_test\n{seq}\n")

def deleak_uniprot_seq_splits():
    path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/unique_sequences_uniprot_id"
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val = pd.read_csv(os.path.join(path, "val.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))

    # consider only positive samples for deleaking
    train = train[train["label"] == 1]
    val = val[val["label"] == 1]
    test = test[test["label"] == 1]

    # Deleaking based on cd-hit-2d results
    print("Deleaking uniprot sequence splits based on cd-hit-2d results...\n")

    print(f"(POSITIVES) Before deleaking, train has {len(train)} entries, val has {len(val)} entries, test has {len(test)} entries.\n")

    cd_hit_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/unique_sequences_uniprot_id/cd_hit"
    train = remove_similar_sequences(train, "train", "val", cd_hit_path)
    train = remove_similar_sequences(train, "train", "test", cd_hit_path)
    test = remove_similar_sequences(test, "test", "val", cd_hit_path)

    print(f"(POSITIVES) After deleaking, train has {len(train)} entries, val has {len(val)} entries, test has {len(test)} entries.\n")

    # sample negatives after deleaking
    neg_train = sample_negatives(train, split="train", n_samples=len(train), self_interactions=True)
    neg_val = sample_negatives(val, split="val", n_samples=len(val), self_interactions=True)
    neg_test = sample_negatives(test, split="test", n_samples=len(test), self_interactions=True)

    train = pd.concat([train, neg_train], ignore_index=True)
    val = pd.concat([val, neg_val], ignore_index=True)
    test = pd.concat([test, neg_test], ignore_index=True)

    train.to_csv(os.path.join(path, "deleak_cd_hit", "train.csv"), index=False)
    val.to_csv(os.path.join(path, "deleak_cd_hit", "val.csv"), index=False)
    test.to_csv(os.path.join(path, "deleak_cd_hit", "test.csv"), index=False)

def completely_balanced_splits():
    path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/full_uniprot_sequences/half_balanced"
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val = pd.read_csv(os.path.join(path, "val.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))

    # consider only positive samples for deleaking
    val = val[val["label"] == 1]
    test = test[test["label"] == 1]

    # val and test should have as many self interacting in positive as in negative -> remove from positive set
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

    val = pd.concat([val, neg_val], ignore_index=True)
    test = pd.concat([test, neg_test], ignore_index=True)

    # save splits
    out_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/full_uniprot_sequences/completely_balanced"
    train.to_csv(os.path.join(out_path, "train.csv"), index=False)
    val.to_csv(os.path.join(out_path, "val.csv"), index=False)
    test.to_csv(os.path.join(out_path, "test.csv"), index=False)
    


if __name__ == "__main__":
    # create_full_uniprot_seq_splits()

    # add_sequences_from_uniparc()

    # generate_fasta_splits()

    # deleak_uniprot_seq_splits()

    completely_balanced_splits()