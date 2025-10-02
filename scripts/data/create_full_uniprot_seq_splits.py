import pandas as pd
import os

from create_dataset import sample_negatives

def replace_sequences(df, seq_dict):
    for idx, row in df.iterrows():
        rec_uniprot = row["entry"].split("--")[0].split("_")[-1]
        lig_uniprot = row["entry"].split("--")[1].split("_")[-1]

        if rec_uniprot == "B2UQL7" or lig_uniprot == "B2UQL7":
            print(f"rec seq in df at {idx} is: {df.at[idx, 'rec_seq']}")
            print(f"lig seq in df at {idx} is: {df.at[idx, 'lig_seq']}")
            print(f"dict entry is: {seq_dict['B2UQL7']}")

        if rec_uniprot in seq_dict.keys():
            df.at[idx, "rec_seq"] = seq_dict[rec_uniprot]
        else:
            print(f"Missing {rec_uniprot}")

        if lig_uniprot in seq_dict.keys():
            df.at[idx, "lig_seq"] = seq_dict[lig_uniprot]
        else:
            print(f"Missing {lig_uniprot}")
        
    return df

if __name__ == "__main__":
    splits_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit"
    train_df = pd.read_csv(os.path.join(splits_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_path, "test.csv"))

    train_df = train_df[train_df["label"] == 1]
    val_df = val_df[val_df["label"] == 1]
    test_df = test_df[test_df["label"] == 1]

    uniprot_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/unique_sequences_uniprot_id"

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

    train_df = sample_negatives(train_df, split="train", n_samples=len(train_df))
    val_df = sample_negatives(val_df, split="val", n_samples=len(val_df))
    test_df = sample_negatives(test_df, split="test", n_samples=len(test_df))

    train_df.to_csv(os.path.join(uniprot_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(uniprot_path, "val.csv"), index=False)
    test_df.to_csv(os.path.join(uniprot_path, "test.csv"), index=False)

    