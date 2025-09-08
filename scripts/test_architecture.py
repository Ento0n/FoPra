import pandas as pd
import torch
import hashlib
import os
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from models.linearFC import SimpleInteractionNet
from models.baseline_fc_conv import baseline2d
from models.richoux_fc import RichouxInteractionNet
from functools import partial


# wrap a pandas DataFrame of sequence pairs into a torch Dataset
class SequencePairDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'receptor_seq': row['receptor_seq'],
            'ligand_seq': row['ligand_seq'],
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

def setup(cache_dir, train_csv, val_csv, test_csv, residue, one_hot, kernel_size, wand_mode="online"):
    #################__Torch device__################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #################__WandB__######################

    wandb.login(key="")

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Username
        entity="tonitobacco",
        # Project
        project="FoPra",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.0001,
            "architecture": "baselineFC",
            "dataset": "Pinder_fully_deleaked",
            "batch_size": 4,
            "epochs": 5,
        },
        # mode
        mode=wand_mode,
    )

    ##################__Dataset__########################

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # instantiate datasets
    train_dataset = SequencePairDataset(train_df)
    val_dataset   = SequencePairDataset(val_df)
    test_dataset  = SequencePairDataset(test_df)

    # Compute fixed global pad lengths for one-hot case to keep model input size constant
    pad_rec_len = None
    pad_lig_len = None
    if one_hot and not residue:
        rec_max_train = train_df["receptor_seq"].str.len().max()
        rec_max_val   = val_df["receptor_seq"].str.len().max()
        rec_max_test  = test_df["receptor_seq"].str.len().max()
        lig_max_train = train_df["ligand_seq"].str.len().max()
        lig_max_val   = val_df["ligand_seq"].str.len().max()
        lig_max_test  = test_df["ligand_seq"].str.len().max()
        pad_rec_len = int(max(rec_max_train, rec_max_val, rec_max_test))
        pad_lig_len = int(max(lig_max_train, lig_max_val, lig_max_test))
        print(f"Using fixed one-hot pad lengths: receptor={pad_rec_len}, ligand={pad_lig_len}")

    collate = partial(collate_fn, cache_dir=cache_dir, residue=residue, one_hot=one_hot, kernel_size=kernel_size, pad_rec_len=pad_rec_len, pad_lig_len=pad_lig_len)

    train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_dataset,   batch_size=run.config.batch_size,               collate_fn=collate)
    test_loader  = DataLoader(test_dataset,  batch_size=run.config.batch_size,               collate_fn=collate)

    return device, run, train_loader, val_loader, test_loader

def aa_one_hot(seq: str) -> torch.Tensor:
    """
    Convert an amino acid sequence into a one-hot encoded tensor.
    
    Args:
        seq (str): Amino acid sequence (e.g. "MTEYK").
    
    Returns:
        torch.Tensor: Shape (L, 20), where L = len(seq).
    """
    # Convert sequence to indices
    idxs = [aa_to_idx[aa] for aa in seq]
    idxs = torch.tensor(idxs, dtype=torch.long)

    # One-hot encode
    one_hot = F.one_hot(idxs, num_classes=len(AA_ALPHABET))
    return one_hot.float()  # often float is convenient for models

# custom collate function to handle variable-length sequences
def collate_fn(batch, cache_dir, residue, one_hot, kernel_size, pad_rec_len, pad_lig_len):
    labels = torch.stack([b["label"] for b in batch])
    recs = []
    ligs = []

    if not one_hot:
        for b in batch:
            # generate MD5 key and retrieve embeddings
            # receptor
            key = hashlib.md5(b["receptor_seq"].encode()).hexdigest()
            recs.append(torch.load(
                os.path.join(cache_dir, f"{key}.pt")
            )["embedding"])
            # ligand
            key = hashlib.md5(b["ligand_seq"].encode()).hexdigest()
            ligs.append(torch.load(
                os.path.join(cache_dir, f"{key}.pt")
            )["embedding"])

        # pad sequences to the maximum length in the batch, consider kernel size
        if residue:
            # Compute max lengths and round up to next multiple of kernel_size
            max_rec_len = max(rec.size(0) for rec in recs)
            max_lig_len = max(lig.size(0) for lig in ligs)

            # Round up to nearest multiple of `kernel_size`
            padded_rec_len = int(math.ceil(max_rec_len / kernel_size) * kernel_size)
            padded_lig_len = int(math.ceil(max_lig_len / kernel_size) * kernel_size)

            # Pad sequences to the new lengths
            recs = [F.pad(rec, (0, 0, 0, padded_rec_len - rec.size(0))) for rec in recs]
            ligs = [F.pad(lig, (0, 0, 0, padded_lig_len - lig.size(0))) for lig in ligs]

        recs = torch.stack(recs)
        ligs = torch.stack(ligs)
    
    else:
        recs = [aa_one_hot(b["receptor_seq"]) for b in batch] # List of (Lr, A)
        ligs = [aa_one_hot(b["ligand_seq"]) for b in batch] # List of (Ll, A)

        # Pad sequences to the global longest sequence
        recs = [F.pad(rec, (0, 0, 0, pad_rec_len - rec.size(0))) for rec in recs] # (Lr, A) -> (padded_Lr, A)
        ligs = [F.pad(lig, (0, 0, 0, pad_lig_len - lig.size(0))) for lig in ligs] # (Ll, A) -> (padded_Ll, A)

        # flatten the one hot encoding and stack batch in 1 tensor
        recs = torch.stack([torch.flatten(rec) for rec in recs]) # (B, A * padded_Lr)
        ligs = torch.stack([torch.flatten(lig) for lig in ligs]) # (B, A * padded_Ll)

        print(recs.shape, ligs.shape)

    return recs, ligs, labels

#######################################__Model__##################################################

def setup_model(train_loader, device, run, residue):
    # 2. Setup device, model, optimizer, and loss function
    if residue:
        rec_emb_sample, _,  _ = next(iter(train_loader))
        embed_dim = rec_emb_sample.size(2)  # Assuming the embedding dimension is the second dimension
    else:
        rec_emb_sample, lig_emb_sample, _ = next(iter(train_loader))
        embed_dim = rec_emb_sample.size(1) + lig_emb_sample.size(1)  # Assuming the embedding dimension is the second dimension

    # Initialize the model
    if residue:
        model = baseline2d(embed_dim).to(device)
    else:
        model = SimpleInteractionNet(embed_dim).to(device)
        # model = RichouxInteractionNet(embed_dim).to(device)

    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)

    if residue:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Give model to WandB
    wandb.watch(model, log="all", log_freq=100)

    return model, optimizer, criterion


def train(device, run, model, optimizer, criterion, train_loader, val_loader, residue):
    # 3. Training loop
    for i in range(run.config.epochs):

        model.train()
        total_loss = 0.0
        for rec_emb, lig_emb, labels in train_loader:
            rec_emb, lig_emb, labels = rec_emb.to(device), lig_emb.to(device), labels.to(device)
            logits = model(rec_emb, lig_emb)

            if residue:
                labels = labels.float()  # Convert labels to float for BCELoss

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {i} training loss: {total_loss / len(train_loader):.4f}")
        # log train loss to wandb
        run.log({"train_loss": total_loss / len(train_loader)}, step=i)

        # 4. Validation
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for rec_emb, lig_emb, labels in val_loader:
                rec_emb, lig_emb, labels = rec_emb.to(device), lig_emb.to(device), labels.to(device)
                logits = model(rec_emb, lig_emb)

                if residue:
                    labels = labels.float()  # Convert labels to float for BCELoss

                # Loss
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Accuracy
                if residue: 
                    preds = (logits > 0.5).float() # (B)
                else:
                    preds = logits.argmax(dim=1) # (B, 2)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation accuracy: {correct/total*100:.2f}%, Validation loss: {avg_val_loss:.4f}")
        # log validation metrics to wandb
        run.log({"val_accuracy": correct/total, "val_loss": avg_val_loss}, step=i)

def test(device, model, test_loader, residue):
    # 5. Test the model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for rec_emb, lig_emb, labels in test_loader:
            rec_emb, lig_emb, labels = rec_emb.to(device), lig_emb.to(device), labels.to(device)

            if residue:
                labels = labels.float()  # Convert labels to float for BCELoss

            logits = model(rec_emb, lig_emb)

            # Accuracy
            if residue:
                preds = (logits > 0.5).float()
            else:
                preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test accuracy: {correct/total*100:.2f}%")


def main():
    # global
    global AA_ALPHABET, aa_to_idx

    # Define the amino acid alphabet (can adapt if you want gaps, X, etc.)
    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXB"
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}

    # Arguments
    residue = False
    one_hot = True
    if residue:
        embedding_type = "residue"
    else:
        embedding_type = "mean"

    train_csv = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/train.csv"
    val_csv = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/val.csv"
    test_csv = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/test.csv"
    cache_dir = f"/nfs/scratch/pinder/negative_dataset/my_repository/embeddings/sequence/ESM3/{embedding_type}"

    kernel_size = 2  # Default kernel size, can be adjusted
    wandb_mode = "disabled"  # Change to "online" to enable WandB logging, "disabled" to disable it


    # PIPELINE
    device, run, train_loader, val_loader, test_loader = setup(cache_dir, train_csv, val_csv, test_csv, residue, one_hot, kernel_size, wandb_mode)
    model, optimizer, criterion = setup_model(train_loader, device, run, residue)
    train(device, run, model, optimizer, criterion, train_loader, val_loader, residue)
    test(device, model, test_loader, residue)

if __name__ == "__main__":
    main()


