#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import sys

import torch


def md5_id(seq):
    return hashlib.md5(seq.encode()).hexdigest()


def load_embedding(cache_dir, seq, protein_dim):
    key = md5_id(seq)
    path = os.path.join(cache_dir, f"{key}.pt")
    if not os.path.exists(path):
        return None, key, f"missing embedding: {path}"
    data = torch.load(path, map_location="cpu")
    if "embedding" not in data:
        return None, key, f"missing 'embedding' in {path}"
    emb = data["embedding"]
    if emb.ndim != 2:
        return None, key, f"expected per-residue embedding (L, D), got shape {tuple(emb.shape)} in {path}"
    if protein_dim is not None and emb.shape[1] != protein_dim:
        return None, key, f"expected embedding dim {protein_dim}, got {emb.shape[1]} in {path}"
    return emb, key, None


def prepare_split(csv_path, cache_dir, out_dir, split_name, protein_dim, skip_missing):
    os.makedirs(out_dir, exist_ok=True)
    interactions_path = os.path.join(out_dir, f"{split_name}_interaction.tsv")
    dictionary_dir = os.path.join(out_dir, f"{split_name}_dictionary")
    os.makedirs(dictionary_dir, exist_ok=True)
    dictionary_path = os.path.join(dictionary_dir, "protein_dictionary.pt")

    proteins = {}
    missing = 0
    total = 0
    with open(csv_path, newline="") as f_in, open(interactions_path, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        required = {"receptor_seq", "ligand_seq", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing required columns: {sorted(required)}")

        writer = csv.writer(f_out, delimiter="\t")
        for row in reader:
            total += 1
            rec_seq = row["receptor_seq"]
            lig_seq = row["ligand_seq"]
            label = row["label"]

            rec_emb, rec_id, rec_err = load_embedding(cache_dir, rec_seq, protein_dim)
            lig_emb, lig_id, lig_err = load_embedding(cache_dir, lig_seq, protein_dim)

            if rec_err or lig_err:
                missing += 1
                if skip_missing:
                    continue
                errors = [e for e in [rec_err, lig_err] if e]
                raise RuntimeError("; ".join(errors))

            if rec_id not in proteins:
                proteins[rec_id] = rec_emb
            if lig_id not in proteins:
                proteins[lig_id] = lig_emb

            writer.writerow([rec_id, lig_id, label])

    torch.save(proteins, dictionary_path)
    print(f"{split_name}: wrote {len(proteins)} proteins and {total - missing} interactions")
    if missing:
        print(f"{split_name}: skipped {missing} interactions due to missing embeddings")


def main():
    parser = argparse.ArgumentParser(description="Prepare TUnA dictionaries and interaction TSVs from CSV splits.")
    parser.add_argument("--csv_dir", required=True, help="Directory containing train.csv/val.csv/test.csv")
    parser.add_argument("--cache_dir", required=True, help="Embedding cache directory (per-residue embeddings)")
    parser.add_argument("--out_dir", required=True, help="Output directory for TUnA files")
    parser.add_argument("--protein_dim", type=int, default=640, help="Expected embedding dimension")
    parser.add_argument("--skip_missing", action="store_true", help="Skip pairs with missing embeddings instead of erroring")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to process")
    args = parser.parse_args()

    for split in args.splits:
        csv_path = os.path.join(args.csv_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {split}: missing {csv_path}")
            continue
        prepare_split(csv_path, args.cache_dir, args.out_dir, split, args.protein_dim, args.skip_missing)


if __name__ == "__main__":
    main()
