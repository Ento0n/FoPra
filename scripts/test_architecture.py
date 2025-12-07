import pandas as pd
import numpy as np
import torch
import hashlib
import os
import sys
import math
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from models.linearFC import LinearFC
from functools import partial
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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

# Torch Dataset considering test classes
class SequencePairDatasetWithClasses(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'receptor_seq': row['receptor_seq'],
            'ligand_seq': row['ligand_seq'],
            'label': torch.tensor(row['label'], dtype=torch.long),
            'class': row['class']
        }

def setup(path, cache_dir, train_csv, val_csv, test_csv, residue, one_hot, kernel_size, wand_mode="online", epochs=5, model_name="linearFC", batch_size=4, judith_test_csv=None):
    #################__Torch device__################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "\n")

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
            "architecture": model_name,
            "dataset": path,
            "batch_size": batch_size,
            "epochs": epochs,
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
    test_dataset  = SequencePairDatasetWithClasses(test_df)

    if judith_test_csv is not None:
        judith_test_df = pd.read_csv(judith_test_csv)
        judith_test_dataset = SequencePairDataset(judith_test_df)

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

        if judith_test_csv is not None:
            judith_rec_max_test = judith_test_df["receptor_seq"].str.len().max()
            judith_lig_max_test = judith_test_df["ligand_seq"].str.len().max()
        else:
            # fallback to 0 if no judith test set provided
            judith_rec_max_test = 0
            judith_lig_max_test = 0

        pad_rec_len = int(max(rec_max_train, rec_max_val, rec_max_test, judith_rec_max_test))
        pad_lig_len = int(max(lig_max_train, lig_max_val, lig_max_test, judith_lig_max_test))
        print(f"Using fixed one-hot pad lengths: receptor={pad_rec_len}, ligand={pad_lig_len}\n")

    collate = partial(collate_fn, test=False, cache_dir=cache_dir, residue=residue, one_hot=one_hot, kernel_size=kernel_size, pad_rec_len=pad_rec_len, pad_lig_len=pad_lig_len)
    collate_test = partial(collate_fn, test=True, cache_dir=cache_dir, residue=residue, one_hot=one_hot, kernel_size=kernel_size, pad_rec_len=pad_rec_len, pad_lig_len=pad_lig_len)

    train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_dataset,   batch_size=run.config.batch_size,               collate_fn=collate)
    test_loader  = DataLoader(test_dataset,  batch_size=run.config.batch_size,               collate_fn=collate_test)

    if judith_test_csv is not None:
        judith_test_loader  = DataLoader(judith_test_dataset,  batch_size=run.config.batch_size,               collate_fn=collate)
    else:
        judith_test_loader = None

    return device, run, train_loader, val_loader, test_loader, judith_test_loader

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
def collate_fn(batch, test, cache_dir, residue, one_hot, kernel_size, pad_rec_len, pad_lig_len):
    labels = torch.stack([b["label"] for b in batch])
    recs = []
    ligs = []

    if not one_hot:
        flag = False
        for b in batch:
            # generate MD5 key and retrieve embeddings
            # receptor
            key = hashlib.md5(b["receptor_seq"].encode()).hexdigest()
            if os.path.exists(os.path.join(cache_dir, f"{key}.pt")):
                recs.append(torch.load(
                    os.path.join(cache_dir, f"{key}.pt")
                )["embedding"].squeeze(0))  # remove batch dim
            else:
                print(f"Embedding for receptor sequence with key {key} not found in cache, but continue. Seq: {b['receptor_seq']}")
                flag = True
                continue

            # ligand
            key = hashlib.md5(b["ligand_seq"].encode()).hexdigest()
            if os.path.exists(os.path.join(cache_dir, f"{key}.pt")):
                ligs.append(torch.load(
                    os.path.join(cache_dir, f"{key}.pt")
                )["embedding"].squeeze(0))  # remove batch dim
            else:
                print(f"Embedding for ligand sequence with key {key} not found in cache, but continue. Seq: {b['ligand_seq']}")
                flag = True
                continue
        
        # if any embedding was missing, return None to skip this batch
        if flag:
            if test:
                return None, None, None, None
            else:
                return None, None, None

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
    
    # for test set, also return classes
    if test:
        # encode to integer classes for torch tensor
        classes = []
        for b in batch:
            cls = b["class"]
            if cls == "self":
                classes.append(1)
            elif cls == "non-self":
                classes.append(2)
            elif cls == "undefined":
                classes.append(3)
        classes = torch.tensor(classes, dtype=torch.long)
        
        return recs, ligs, labels, classes

    return recs, ligs, labels

#######################################__Model__##################################################

def setup_model(train_loader, device, run, residue, model_name):
    # 2. Setup device, model, optimizer, and loss function
    if residue:
        rec_emb_sample, _,  _ = next(iter(train_loader))
        embed_dim = rec_emb_sample.size(2)  # Assuming the embedding dimension is the third dimension
    else:
        rec_emb_sample, lig_emb_sample, _ = next(iter(train_loader))
        embed_dim = rec_emb_sample.size(1) + lig_emb_sample.size(1)  # Assuming the embedding dimension is the second dimension

    if model_name == "LinearFC":
        model = LinearFC(embed_dim).to(device)
    elif model_name == "baseline2d":
        from models.baseline_fc_conv import baseline2d
        model = baseline2d(embed_dim).to(device)

    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)

    # Binary Cross Entropy Loss for binary classification
    criterion = nn.BCEWithLogitsLoss()  # more stable than BCELoss with separate sigmoid

    # Give model to WandB
    wandb.watch(model, log="all", log_freq=100)

    return model, optimizer, criterion


def train(device, run, model, optimizer, criterion, train_loader, val_loader, training_limit):
    # give info that training size is limited
    if training_limit:
        print(f"*** Training is limited to {training_limit} batches per epoch for debugging purposes ***\n")

    # Training loop
    for i in range(run.config.epochs):

        model.train()
        total_loss = 0.0
        for idx, (rec_emb, lig_emb, labels) in enumerate(train_loader):
            if rec_emb is None or lig_emb is None or labels is None:
                # skip this batch due to missing embeddings
                continue

            # early stopping for debugging with limited training samples
            if training_limit and idx >= training_limit:
                break

            rec_emb, lig_emb, labels = rec_emb.to(device), lig_emb.to(device), labels.to(device)
            logits = model(rec_emb, lig_emb)

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
                if rec_emb is None or lig_emb is None or labels is None:
                    # skip this batch due to missing embeddings
                    continue

                rec_emb, lig_emb, labels = rec_emb.to(device), lig_emb.to(device), labels.to(device)
                logits = model(rec_emb, lig_emb)

                labels = labels.float()  # Convert labels to float for BCELoss

                # Loss
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Accuracy
                probs = torch.sigmoid(logits) # (B)
                preds = (probs > 0.5).float() # (B)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation accuracy: {correct/total*100:.2f}%, Validation loss: {avg_val_loss:.4f}")
        # log validation metrics to wandb
        run.log({"val_accuracy": correct/total, "val_loss": avg_val_loss}, step=i)
    
    # new line after training
    print("\n")


def test(device, model, test_loader):
    model.eval()

    # general
    correct, total = 0, 0
    neg, pos = 0, 0

    # self
    correct_self, total_self = 0, 0
    self_preds, self_labels = [], []
    self_neg, self_pos = 0, 0

    # non-self
    correct_nonself, total_nonself = 0, 0
    nonself_preds, nonself_labels = [], []
    nonself_neg, nonself_pos = 0, 0

    # undefined
    correct_undefined, total_undefined = 0, 0
    undefined_preds, undefined_labels = [], []
    undefined_neg, undefined_pos = 0, 0

    # all
    all_probs, all_preds, all_labels, all_classes = [], [], [], []
    falsy_idxs = []
    with torch.no_grad():
        for idx, (rec_emb, lig_emb, labels, classes) in enumerate(test_loader):
            if rec_emb is None or lig_emb is None or labels is None:
                # skip this batch due to missing embeddings
                falsy_idxs.extend([idx*4, idx*4+1, idx*4+2, idx*4+3])  # approximate batch size of 4
                continue

            rec_emb, lig_emb, labels = rec_emb.to(device), lig_emb.to(device), labels.to(device)

            labels = labels.float()  # Convert labels to float for BCELoss

            logits = model(rec_emb, lig_emb)

            # Accuracy
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # general
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            neg += (labels == 0).sum().item()
            pos += (labels == 1).sum().item()

            # move pred and labels to cpu for class-specific accuracy
            preds = preds.cpu()
            labels = labels.cpu()

            # self
            correct_self += ((preds == labels) & (classes == 1)).sum().item()
            total_self += (classes == 1).sum().item()
            self_preds.extend(preds[classes == 1].cpu().numpy())
            self_labels.extend(labels[classes == 1].cpu().numpy())
            self_neg += (labels[classes == 1] == 0).sum().item()
            self_pos += (labels[classes == 1] == 1).sum().item()

            # non-self
            correct_nonself += ((preds == labels) & (classes == 2)).sum().item()
            total_nonself += (classes == 2).sum().item()
            nonself_preds.extend(preds[classes == 2].cpu().numpy())
            nonself_labels.extend(labels[classes == 2].cpu().numpy())
            nonself_neg += (labels[classes == 2] == 0).sum().item()
            nonself_pos += (labels[classes == 2] == 1).sum().item()

            # undefined
            correct_undefined += ((preds == labels) & (classes == 3)).sum().item()
            total_undefined += (classes == 3).sum().item()
            undefined_preds.extend(preds[classes == 3].cpu().numpy())
            undefined_labels.extend(labels[classes == 3].cpu().numpy())
            undefined_neg += (labels[classes == 3] == 0).sum().item()
            undefined_pos += (labels[classes == 3] == 1).sum().item()

            # Collect all probs, preds and labels for confusion matrix
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_classes.extend(classes.cpu().numpy())

    print(f"General test accuracy: {correct/total*100:.2f}% ({total} samples)")
    print(f"Total negative samples: {neg}, Total positive samples: {pos}")
    print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds, labels=[0,1])}\n")

    if total_self > 0:
        print(f"Self-interaction test accuracy: {correct_self/total_self*100:.2f}% ({total_self} samples)")
        print(f"Self negative samples: {self_neg}, Self positive samples: {self_pos}")
        print(f"Confusion Matrix:\n{confusion_matrix(self_labels, self_preds, labels=[0,1])}\n")
    else:
        print("No samples in self class.\n")

    if total_nonself > 0:
        print(f"Non-self-interaction test accuracy: {correct_nonself/total_nonself*100:.2f}% ({total_nonself} samples)")
        print(f"Non-self negative samples: {nonself_neg}, Non-self positive samples: {nonself_pos}")
        print(f"Confusion Matrix:\n{confusion_matrix(nonself_labels, nonself_preds, labels=[0,1])}\n")
    else:
        print("No samples in non-self class.\n")

    if total_undefined > 0:
        print(f"Undefined class test accuracy: {correct_undefined/total_undefined*100:.2f}% ({total_undefined} samples)")
        print(f"Undefined negative samples: {undefined_neg}, Undefined positive samples: {undefined_pos}")
        print(f"Confusion Matrix:\n{confusion_matrix(undefined_labels, undefined_preds, labels=[0,1])}\n")
    else:
        print("No samples in undefined class.\n")

    all_probs = np.round(np.array(all_probs, dtype=np.float32), 3).tolist()
    all_probs = [round(x, 3) for x in all_probs]

    return all_probs,all_preds, all_labels, all_classes, falsy_idxs

def test_judith_gold(device, model, judith_test_loader):
    print("Testing on Judith Gold Standard dataset...\n")
    model.eval()

    correct, total = 0, 0
    neg, pos = 0, 0

    with torch.no_grad():
        all_probs, all_preds, all_labels = [], [], []
        falsy_idxs = []
        for idx, (rec_emb, lig_emb, labels) in enumerate(judith_test_loader):
            if rec_emb is None or lig_emb is None or labels is None:
                # skip this batch due to missing embeddings
                falsy_idxs.extend([idx*4, idx*4+1, idx*4+2, idx*4+3])  # approximate batch size of 4
                continue

            rec_emb, lig_emb, labels = rec_emb.to(device), lig_emb.to(device), labels.to(device)

            labels = labels.float()  # Convert labels to float for BCELoss

            logits = model(rec_emb, lig_emb)

            # Accuracy
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # general
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            neg += (labels == 0).sum().item()
            pos += (labels == 1).sum().item()

            # Collect all probs, preds and labels for confusion matrix
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.round(np.array(all_probs, dtype=np.float32), 3).tolist()
    all_probs = [round(x, 3) for x in all_probs]

    print(f"Judith Gold Standard test accuracy: {correct/total*100:.2f}% ({total} samples)")
    print(f"Total negative samples: {neg}, Total positive samples: {pos}")
    print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds, labels=[0,1])}\n")

    return all_probs, all_preds, all_labels, falsy_idxs

def handle_random_forest(train_csv, test_csv):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print("Training Random Forest Classifier on one-hot encoded sequences...\n")

    # Compute fixed global pad lengths for one-hot case to keep model input size constant
    rec_max_train = train_df["receptor_seq"].str.len().max()
    rec_max_test  = test_df["receptor_seq"].str.len().max()
    lig_max_train = train_df["ligand_seq"].str.len().max()
    lig_max_test  = test_df["ligand_seq"].str.len().max()
    pad_rec_len = int(max(rec_max_train, rec_max_test))
    pad_lig_len = int(max(lig_max_train, lig_max_test))
    print(f"Using fixed one-hot pad lengths: receptor={pad_rec_len}, ligand={pad_lig_len}\n")

    def extract_features(df):
        recs = [aa_one_hot(aa) for aa in df["receptor_seq"]] # List of (Lr, A)
        ligs = [aa_one_hot(aa) for aa in df["ligand_seq"]] # List of (Ll, A)

        # Pad sequences to the global longest sequence
        recs = [F.pad(rec, (0, 0, 0, pad_rec_len - rec.size(0))) for rec in recs] # (Lr, A) -> (padded_Lr, A)
        ligs = [F.pad(lig, (0, 0, 0, pad_lig_len - lig.size(0))) for lig in ligs] # (Ll, A) -> (padded_Ll, A)

        # flatten the one hot encoding and stack batch in 1 tensor
        recs = torch.stack([torch.flatten(rec) for rec in recs]) # (df_size, A * padded_Lr)
        ligs = torch.stack([torch.flatten(lig) for lig in ligs]) # (df_size, A * padded_Ll)

        features = torch.cat([recs, ligs], dim=1) # (df_size, A * (padded_Lr + padded_Ll))

        return features

    X_train = extract_features(train_df).numpy()
    y_train = train_df["label"].values

    X_test = extract_features(test_df).numpy()
    y_test = test_df["label"].values

    clf = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True, max_samples=0.2)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

@torch.no_grad()
def lrp_simple_interaction_net(model, rec_emb, lig_emb, start_from, target, rule_first, rule_hidden, eps: float = 1e-6, normalize: bool = False):

    # cancels out dropout
    model.eval()

    # Build iput and remember split point
    x = torch.cat([rec_emb, lig_emb], dim=1)  # (B, D)
    d_rec = rec_emb.size(1)

    # Collect Linear layers in order
    linears = [m for m in model.fc if isinstance(m, nn.Linear)]
    if len(linears) < 1:
        raise RuntimeError("No Linear layers found in model.fc (required for LRP).")

    # Forward pass while caching inputs to each Linear (activations before that Linear)
    h = x
    activations = []
    for layer in model.fc:
        if isinstance(layer, nn.Linear):
            activations.append(h)
        h = layer(h)
    
    # Final output
    # Keep probability as (B, 1) to avoid broadcasting issues in LRP
    prob = h  # after sigmoid: (B, 1)

    # before sigmoid: a_last @ W_last^T + b_last
    a_last = activations[-1]  # (B, H)
    W_last = linears[-1].weight  # (1, H)
    b_last = linears[-1].bias  # (1)
    logit = a_last @ W_last.t() + b_last  # (B, 1)

    # Choose starting relevance R^L
    if start_from == "logit":
        R = logit.clone()
    elif start_from == "prob":
        if target == "pos":
            R = prob.clone() # (B, 1)
        elif target == "neg":
            R = 1.0 - prob # (B, 1)
        else:
            raise ValueError("target must be 'pos' or 'neg'")
    else:
        raise ValueError("start_from must be 'prob' or 'logit'")
    
    # Local propagation rules for a Linear layer
    def back_linear_epsilon(a: torch.Tensor, lin: nn.Linear, R: torch.Tensor, eps: float):
        # Shapes: a (B,in), weight (out,in), bias (out), R (B,out)
        z = a @ lin.weight.t() + lin.bias                      # (B,out)
        stabilizer = eps * torch.where(z >= 0, 1.0, -1.0)      # signed epsilon
        s = R / (z + stabilizer)                               # (B,out)
        # R_in = a * (s @ W)   (efficient implementation of sum_j (a_i * w_ij * R_j / z_j))
        return a * (s @ lin.weight)                            # (B,in)

    def back_linear_zplus(a: torch.Tensor, lin: nn.Linear, R: torch.Tensor, eps: float):
        Wp = torch.clamp(lin.weight, min=0.0)                  # (out,in)
        ap = torch.clamp(a, min=0.0)                           # (B,in)  (assumes ReLU in previous layers)
        z = ap @ Wp.t() + eps                                  # (B,out) stabilizer keeps denom > 0
        s = R / z                                              # (B,out)
        return ap * (s @ Wp)                                   # (B,in)

    # Decide per-layer rules: first layer often uses epsilon; hidden layers z+
    rules = [rule_first] + [rule_hidden] * (len(linears) - 1)

    # Backward LRP pass through Linear layers in reverse order
    for a, lin, rule in zip(reversed(activations), reversed(linears), reversed(rules)):
        if rule == "epsilon":
            R = back_linear_epsilon(a, lin, R, eps)
        elif rule == "zplus":
            R = back_linear_zplus(a, lin, R, eps)
        else:
            raise ValueError("rule must be 'epsilon' or 'zplus'")

    R_in = R  # (B, input_dim)

    # Optional per-sample normalization (useful for visualization comparability)
    if normalize:
        denom = R_in.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
        R_in = R_in / denom

    # Split into receptor vs ligand relevance
    R_rec = R_in[:, :d_rec]
    R_lig = R_in[:, d_rec:]

    return R_in, R_rec, R_lig, {"prob": prob.squeeze(-1), "logit": logit.squeeze(-1)}

def exec_lrp_simple_interaction_net(model, device, test_loader, one_hot):
    model.eval()

    total_sum = None
    rec_sum = None
    lig_sum = None
    total_count = 0

    # Optional class-conditional aggregation
    pos_sum = None
    neg_sum = None
    pos_count = 0
    neg_count = 0

    d_rec = None  # will be set from first batch

    with torch.no_grad():
        for rec_emb, lig_emb, labels in test_loader:
            if rec_emb is None or lig_emb is None or labels is None:
                # skip this batch due to missing embeddings
                continue

            rec_emb, lig_emb, labels = rec_emb.to(device), lig_emb.to(device), labels.to(device).float()
            if d_rec is None:
                d_rec = rec_emb.size(1)

            R_in, R_rec, R_lig, info = lrp_simple_interaction_net(
                model, rec_emb, lig_emb,
                start_from="prob",
                target="pos",
                rule_first="epsilon",
                rule_hidden="zplus",
                eps=1e-6,
                normalize=False
            )

            # Use absolute relevance for aggregation (recommended)
            Rin_abs = R_in.abs()
            Rrec_abs = R_rec.abs()
            Rlig_abs = R_lig.abs()

            if total_sum is None:
                total_sum = Rin_abs.sum(dim=0)         # (D,)
                rec_sum   = Rrec_abs.sum(dim=0)        # (D_rec,)
                lig_sum   = Rlig_abs.sum(dim=0)        # (D_lig,)
                pos_sum   = torch.zeros_like(total_sum)
                neg_sum   = torch.zeros_like(total_sum)
            else:
                total_sum += Rin_abs.sum(dim=0)
                rec_sum   += Rrec_abs.sum(dim=0)
                lig_sum   += Rlig_abs.sum(dim=0)

            bsz = Rin_abs.size(0)
            total_count += bsz

            # Class-conditional means
            pos_mask = (labels == 1.0).view(-1, 1)  # (B,1)
            neg_mask = (labels == 0.0).view(-1, 1)
            if pos_mask.any():
                pos_sum += (Rin_abs * pos_mask).sum(dim=0)
                pos_count += int(pos_mask.sum().item())
            if neg_mask.any():
                neg_sum += (Rin_abs * neg_mask).sum(dim=0)
                neg_count += int(neg_mask.sum().item())

    # Means
    mean_all = (total_sum / max(total_count, 1)).detach().cpu().numpy()
    mean_rec = (rec_sum   / max(total_count, 1)).detach().cpu().numpy()
    mean_lig = (lig_sum   / max(total_count, 1)).detach().cpu().numpy()

    mean_pos = (pos_sum / max(pos_count, 1)).detach().cpu().numpy() if pos_count > 0 else None
    mean_neg = (neg_sum / max(neg_count, 1)).detach().cpu().numpy() if neg_count > 0 else None
    
    # k highest influences
    k = 3

    rec_idxs = np.argsort(mean_rec)[-k:][::-1]
    lig_idxs = np.argsort(mean_lig)[-k:][::-1]
    rec_vals = mean_rec[rec_idxs]
    lig_vals = mean_lig[lig_idxs]

    print(f"Top-{k} receptor feature indices with highest mean |R_in|:")
    for i in range(k):
        print(f" {i+1}: index {rec_idxs[i]}, mean |R_in|={rec_vals[i]}")
        if one_hot:
            aa_idx = rec_idxs[i] % len(AA_ALPHABET)
            pos_idx = rec_idxs[i] // len(AA_ALPHABET)
            print(f"     -> position {pos_idx}, amino acid '{AA_ALPHABET[aa_idx]}'")
    print("\n")

    print(f"Top-{k} ligand feature indices with highest mean |R_in|:")
    for i in range(k):
        print(f" {i+1}: index {lig_idxs[i]}, mean |R_in|={lig_vals[i]}")
        if one_hot:
            aa_idx = lig_idxs[i] % len(AA_ALPHABET)
            pos_idx = lig_idxs[i] // len(AA_ALPHABET)
            print(f"     -> position {pos_idx}, amino acid '{AA_ALPHABET[aa_idx]}'")
    print("\n")

    # PLOTTING

    out_dir = "/nfs/scratch/pinder/negative_dataset/my_repository/plots"
    os.makedirs(out_dir, exist_ok=True)

    sns.set_theme()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True, sharey=True)

    sns.lineplot(x=np.arange(len(mean_all)), y=mean_all, ax=axes[0], color="blue")
    axes[0].set_title("Total Mean |R_in|")

    sns.lineplot(x=np.arange(len(mean_pos)), y=mean_pos, ax=axes[1], color="green")
    axes[1].set_title("Positive Mean |R_in|")

    sns.lineplot(x=np.arange(len(mean_neg)), y=mean_neg, ax=axes[2], color="red")
    axes[2].set_title("Negative Mean |R_in|")
    axes[2].set_xlabel("Feature Index")

    for ax in axes:
        ax.set_ylabel("Mean |R_in|")
        ax.axvline(x=d_rec, color="black", linestyle=":", label="Receptor/Ligand Split")

    plt.savefig(os.path.join(out_dir, "lrp_total_mean.png"))

        


def main():
    # set seed
    set_seed()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test ML Architecture for Protein Interaction Prediction")
    parser.add_argument('--token', type=str, default=None, required=True, help='Token for WandB login')
    parser.add_argument('--path', type=str, default=None, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--cache_dir', type=str, default=None, help='Path to the embedding cache directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--model', type=str, required=True, help='Model architecture to use')
    parser.add_argument('--random_forest', action='store_true', help='If set, run Random Forest classifier instead of NN')
    parser.add_argument('--lrp', action='store_true', help='If set, perform LRP analysis after testing')
    parser.add_argument('--judith_test', type=str, help='If set, use Judith gold standard test set')
    parser.add_argument('--limit_training', type=int, default=None, help='Limit the number of training samples (for quick tests)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--residue', action='store_true', help='Use residue-level embeddings instead of mean embeddings')
    group.add_argument('--one_hot', action='store_true', help='Use one-hot encoding instead of embeddings')

    args = parser.parse_args()

    # global
    global AA_ALPHABET, aa_to_idx

    # Define the amino acid alphabet (can adapt if you want gaps, X, etc.)
    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXBUZO"  # 23 characters
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    
    # set embedding type
    if args.residue:
        embedding_type = "residue"
    else:
        embedding_type = "mean"
    
    # print encoding type
    if not args.one_hot:
        print(f"Using {embedding_type} embeddings\n")
        print(f"Using embedding cache dir: {args.cache_dir}\n")
    else:
        print("Using one-hot encoding\n")
    
    # print dataset path
    print(f"Using dataset path: {args.path}")

    # dataset paths to csv files
    train_csv = os.path.join(args.path, "train.csv")
    val_csv = os.path.join(args.path, "val.csv")
    test_csv = os.path.join(args.path, "test.csv")

    # !!!!arguments not sure what will happen with in the future!!!!
    kernel_size = 2  # Default kernel size, can be adjusted
    wandb_mode = "disabled"  # Change to "online" to enable WandB logging, "disabled" to disable it

    # execute random forest if specified
    if args.random_forest:
        handle_random_forest(train_csv, test_csv)
        sys.exit(0)

    # Fit training limit to batch size
    batch_size = 4
    if args.limit_training is not None:
        args.limit_training = args.limit_training // batch_size

    # PIPELINE
    device, run, train_loader, val_loader, test_loader, judith_test_loader = setup(args.path, args.cache_dir, train_csv, val_csv, test_csv, args.residue, args.one_hot, kernel_size, wandb_mode, args.epochs, args.model, batch_size, args.judith_test)
    model, optimizer, criterion = setup_model(train_loader, device, run, args.residue, args.model)
    train(device, run, model, optimizer, criterion, train_loader, val_loader, args.limit_training)
    all_probs, all_preds, all_labels, all_classes, falsy_idxs = test(device, model, test_loader)

    # Save predictions to CSV file
    if not args.one_hot:
        encoding = "residue" if args.residue else "mean"
        if encoding == "mean":
            if "ESM3" in args.cache_dir:
                encoding = "mean_ESM3"
            elif "ESM2" in args.cache_dir:
                encoding = "mean_ESM2"
            if "sequence_structure" in args.cache_dir:
                encoding += "_structure"
        if encoding == "residue":
            if "ESM3" in args.cache_dir:
                encoding = "residue_ESM3"
            elif "ESM2" in args.cache_dir:
                encoding = "residue_ESM2"
            if "sequence_structure" in args.cache_dir:
                encoding += "_structure"
    encoding = "one_hot" if args.one_hot else encoding
    test_df = pd.read_csv(os.path.join(args.path, "test.csv"))
    with open(os.path.join(args.path, f"test_predictions_{args.model}_{encoding}.txt"), "w") as f:
        f.write("entry,probability,prediction,label,class,model_name,time_stamp\n")
        falsies = 0
        for idx, row in test_df.iterrows():
            if idx in falsy_idxs:
                f.write("FALSY!!!!!!!!!!!!!!!\n")
                falsies += 1
                continue  # skip entries with missing embeddings
            f.write(f"{row['entry']},{all_probs[idx-falsies]},{all_preds[idx-falsies]},{all_labels[idx-falsies]},{all_classes[idx-falsies]},{args.model},{datetime.now().isoformat(timespec='minutes')}\n")

    # test on judith gold standard if specified
    if args.judith_test:
        all_probs, all_preds, all_labels, falsy_idxs = test_judith_gold(device, model, judith_test_loader)

        with open(os.path.join(args.path, f"judith_gold_standard_predictions_{args.model}_{encoding}.txt"), "w") as f:
            f.write("rec_uniprot,lig_uniprot,probability,prediction,label,model_name,time_stamp\n")

            falsies = 0
            for idx, row in pd.read_csv(args.judith_test).iterrows():
                if idx in falsy_idxs:
                    f.write("FALSY!!!!!!!!!!!!!!!\n")
                    falsies += 1
                    continue  # skip entries with missing embeddings
                f.write(f"{row['receptor_uniprot']},{row['ligand_uniprot']},{all_probs[idx-falsies]},{all_preds[idx-falsies]},{all_labels[idx-falsies]},{args.model},{datetime.now().isoformat(timespec='minutes')}\n")

    # LRP analysis if specified
    if args.lrp:
        print("Executing LRP on test set!!\n")
        exec_lrp_simple_interaction_net(model, device, test_loader, args.one_hot)

if __name__ == "__main__":
    main()
