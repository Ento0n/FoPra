# Do Structural Embeddings Improve Multi-Species PPI Prediction? A Leakage-Controlled Study

This repository contains a protein-protein interaction workflow built around Pinder-derived datasets. It covers three main steps:

- building train/val/test splits with matched negative pairs
- generating ESM2 / ESM3 sequence or sequence+structure embeddings
- training and evaluating interaction models on those splits

The main training entry point is `scripts/test_architecture.py`. It supports three model families:

- `LinearFC`: simple feed-forward baseline on one-hot or mean embeddings
- `baseline2d`: 2D interaction model over residue-level features
- `tuna_timo`: transformer-style interaction model over residue-level features

![Description](plots/overview/Frame%201.png)

## Repository layout

- `datasets/`: generated CSV splits and benchmark variants such as `no_duplicates/deleak_cdhit/fully_balanced`
- `embeddings/`: cached per-sequence `.pt` embeddings keyed by sequence hash
- `scripts/data/`: dataset creation, deleaking, UniProt utilities, and embedding generation
- `scripts/models/`: model definitions
- `plots/`: LRP and analysis figures
- `notebooks/`: exploratory analysis notebooks

## Expected data format

Each dataset split directory contains `train.csv`, `val.csv`, and `test.csv`.

- `train.csv` / `val.csv`: `entry, split, label, receptor_seq, ligand_seq, receptor_path, ligand_path`
- `test.csv`: same fields plus a `class` column (`self`, `non-self`, or `undefined`)

## Environment

There is no single top-level environment file for the whole project, but the code expects at least:

- Python 3.10+
- PyTorch
- pandas / numpy
- scikit-learn
- matplotlib / seaborn
- `wandb`
- `pinder`
- ESM dependencies (`fair-esm` for ESM2, ESM3 + Hugging Face login for ESM3)

## Common commands

Create a dataset split:

```bash
python scripts/data/create_dataset.py \
  --out_path datasets/my_split
```

Generate ESM2 mean embeddings for a split:

```bash
python scripts/data/generate_embeddings.py \
  --path datasets/my_split \
  --model esm2 \
  --out_path embeddings/sequence/ESM2/my_split/mean \
  --token <hf_token>
```

Train and test `LinearFC` on mean embeddings:

```bash
python scripts/test_architecture.py \
  --path datasets/no_duplicates/deleak_cdhit/fully_balanced \
  --model LinearFC \
  --encoding mean \
  --cache_dir embeddings/sequence/ESM2/pinder/mean \
  --epochs 5
```

Train a residue-level model:

```bash
python scripts/test_architecture.py \
  --path datasets/no_duplicates/deleak_cdhit/fully_balanced \
  --model baseline2d \
  --encoding residue \
  --cache_dir embeddings/sequence/ESM3/pinder/residue \
  --epochs 5
```

## Outputs

Training writes predictions back into the selected dataset directory:

- `test_predictions_<model>_<encoding>.txt`

Other outputs:

- Weights & Biases logging is enabled by default in `scripts/test_architecture.py`
- LRP analysis (`--lrp`) writes arrays and figures to `plots/`

## Notes

- Large generated assets in `datasets/` and `embeddings/` are intentionally ignored by Git.
- Most helper shell scripts assume the original cluster paths under `/nfs/scratch/pinder/negative_dataset/...` and are easiest to launch from `scripts/`.
- The repository includes several dataset variants; `datasets/no_duplicates/deleak_cdhit/fully_balanced` is the final benchmark split.
