import os
import pandas as pd


if __name__ == "__main__":
    filtered = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/cd_hit/mine_judith.out"
    judith_dataset_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard"


    with open(filtered, "r") as f:
        filtered_ids = []
        for line in f:
            if line.startswith(">"):
                filtered_ids.append(line[1:].strip())
    

    train_file = os.path.join(judith_dataset_path, "train.csv")
    val_file = os.path.join(judith_dataset_path, "val.csv")
    test_file = os.path.join(judith_dataset_path, "test.csv")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    df = pd.concat([train_df, val_df, test_df])

    # filter entries where both receptor_uniprot and ligand_uniprot are in filtered_ids
    df = df[(df["receptor_uniprot"].isin(filtered_ids)) & (df["ligand_uniprot"].isin(filtered_ids))]

    # filter length of receptor_seq and ligand_seq to be below or equal 2500
    df = df[(df["receptor_seq"].str.len() <= 2500) & (df["ligand_seq"].str.len() <= 2500)]

    out_file = os.path.join(judith_dataset_path, "test_pinder.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved filtered dataset to {out_file}, {len(df)} entries.")
    

