import pandas as pd
import os
import argparse
from collections import Counter

from utils import sample_negatives, add_test_classes


def create_full_uniprot_seq_splits(path: str, fasta_path: str, out_path: str):

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

    train_df = pd.read_csv(os.path.join(path, "train.csv"))
    val_df = pd.read_csv(os.path.join(path, "val.csv"))
    test_df = pd.read_csv(os.path.join(path, "test.csv"))

    train_df = train_df[train_df["label"] == 1]
    val_df = val_df[val_df["label"] == 1]
    test_df = test_df[test_df["label"] == 1]

    # read full fasta into dictionary
    full_seqs = {}
    with open(fasta_path, "r") as f:
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

    # add test classes
    test_df = add_test_classes(df=test_df)

    train_df.to_csv(os.path.join(out_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_path, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_path, "test.csv"), index=False)

def add_sequences_from_uniparc(uniparc_path: str, uniprot_path: str, species_path: str):
    # read tsv file
    with open(uniparc_path, "r") as f:
        extracted_info = {}
        for line in f:
            # skip first line
            if line.startswith("From"):
                continue

            # parse line
            parts = line.strip().split("\t")
            uniprot_id = parts[0]
            sequence = parts[4]
            organism_part_one = parts[2].split(";")[0].split(" ")[0].strip()
            organism_part_two = parts[2].split(";")[0].split(" ")[1].strip() if len(parts[2].split(";")[0].split(" ")) > 1 else ""
            organism = f"{organism_part_one} {organism_part_two}".strip()
            organism_id = parts[3].split(";")[0].strip()

            extracted_info[uniprot_id] = (sequence, organism, organism_id)
    
    print(f"Adding {len(extracted_info)} sequences from uniparc to full_uniprot_sequences.fasta")
    
    # Create uniparc sequences fasta
    with open(uniprot_path, "a") as f:
        for uid, info in extracted_info.items():
            seq, _, _ = info
            f.write(f">{uid}\n")
            f.write(f"{seq}\n")
    
    # Update species file
    species_counter_dict = {}
    with open(species_path, "r") as f:
        for line in f:
            if line.startswith("Species"):
                continue
            
            parts = line.strip().split("\t")
            species_counter_dict[parts[0] + "," + parts[1]] = int(parts[2])
    
    species_counter = Counter(species_counter_dict)
    for uid, info in extracted_info.items():
        _, organism, organism_id = info
        species_counter[organism + "," + organism_id] += 1
    
    with open(species_path + ".new", "w") as f:
        f.write("Species\tTax ID\tCount\n")
        for info, count in species_counter.items():
            species, tax_id = info.split(",")
            f.write(f"{species}\t{tax_id}\t{count}\n")



def create_uniq_uniprot_fasta(path: str, out_path: str):
    print(f"Creating unique uniprot id fasta from splits in {path}\n")
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val = pd.read_csv(os.path.join(path, "val.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))

    # consider only positive samples for fasta
    train = train[train["label"] == 1]
    val = val[val["label"] == 1]
    test = test[test["label"] == 1]

    ids = {}
    for df in [train, val, test]:
        for idx, row in df.iterrows():
            rec_uniprot = row["entry"].split("--")[0].split("_")[-1]
            lig_uniprot = row["entry"].split("--")[1].split("_")[-1]

            ids[rec_uniprot] = row["receptor_seq"]
            ids[lig_uniprot] = row["ligand_seq"]
    
    print(f"Total unique uniprot ids: {len(ids)}")

    # write to fasta
    fasta = os.path.join(out_path, "unique_uniprot_sequences.fasta")
    with open(fasta, "w") as f:
        for uid, seq in ids.items():
            f.write(f">{uid}\n{seq}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="uniprot data processing")
    parser.add_argument('--function', type=str, required=True, help='Function to execute')
    parser.add_argument('--path', type=str, help='Path to read from')
    parser.add_argument('--file_path', type=str, help='Fasta file to read from')
    parser.add_argument('--file2_path', type=str, help='Second file path if needed')
    parser.add_argument('--file3_path', type=str, help='Third file path if needed')
    parser.add_argument('--out_path', type=str, help='Path to save to')
    args = parser.parse_args()

    if args.function == "create_full_uniprot_seq_splits":
        create_full_uniprot_seq_splits(args.path, args.file_path, args.out_path)
    elif args.function == "add_sequences_from_uniparc":
        add_sequences_from_uniparc(args.file_path, args.file2_path, args.file3_path)
    elif args.function == "create_uniq_uniprot_fasta":
        create_uniq_uniprot_fasta(args.path, args.out_path)
    else:
        print(f"Function {args.function} not recognized.")