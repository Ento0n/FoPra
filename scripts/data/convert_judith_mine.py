import pandas as pd
import sys

def create_split_df(pos_file_path, neg_file_path, seqs, split_name):

    # positive interactions
    pos_interactions = []
    with open(pos_file_path, "r") as f:
        for line in f:
            pos_interactions.append(tuple(line.strip().split(" ")))
    
    # negative interactions
    neg_interactions = []
    with open(neg_file_path, "r") as f:
        for line in f:
            neg_interactions.append(tuple(line.strip().split(" ")))
    
    # Create DataFrame
    data = []
    for rec, lig in pos_interactions:
        if rec not in seqs or lig not in seqs:
            continue
        rec_seq = seqs[rec]
        lig_seq = seqs[lig]
        data.append((rec_seq, lig_seq, 1, split_name))
    
    for rec, lig in neg_interactions:
        if rec not in seqs or lig not in seqs:
            continue
        rec_seq = seqs[rec]
        lig_seq = seqs[lig]
        data.append((rec_seq, lig_seq, 0, split_name))
    
    df = pd.DataFrame(data, columns=["receptor_seq", "ligand_seq", "label", "split"])
    return df
    

if __name__ == "__main__":
    seqs_file = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/human_swissprot_oneliner.fasta"

    # Extract sequences from FASTA file
    seqs = {}
    with open(seqs_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                uniprot_id = line[1:].strip()
                seq = next(f).strip()

                # skip too long sequences
                if len(seq) > 2500:
                    print(f"Skipping {uniprot_id} with length {len(seq)}", file=sys.stderr)
                    continue

                seqs[uniprot_id] = seq
    
    # collect train split
    train_pos_file = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/Intra1_pos_rr.txt"
    train_neg_file = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/Intra1_neg_rr.txt"

    train_df = create_split_df(train_pos_file, train_neg_file, seqs, "train")
    train_df.to_csv("/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/train.csv", index=False)

    # collect val split
    val_pos_file = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/Intra0_pos_rr.txt"
    val_neg_file = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/Intra0_neg_rr.txt"

    val_df = create_split_df(val_pos_file, val_neg_file, seqs, "val")
    val_df.to_csv("/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/val.csv", index=False)

    # collect test split
    test_pos_file = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/Intra2_pos_rr.txt"
    test_neg_file = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/Intra2_neg_rr.txt"

    test_df = create_split_df(test_pos_file, test_neg_file, seqs, "test")
    test_df.to_csv("/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/test.csv", index=False)
