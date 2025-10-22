import pandas as pd
import os

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

def add_test_classes():
    test_df_path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/unique_sequences_uniprot_id/deleak_cd_hit"
    test_df = pd.read_csv(os.path.join(test_df_path, "test.csv"))

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
    test_df.to_csv(os.path.join(test_df_path, "test.csv"), index=False)

if __name__ == "__main__":
    # add_labels()

    add_test_classes()