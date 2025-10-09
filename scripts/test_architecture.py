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
from models.linearFC import SimpleInteractionNet
from models.baseline_fc_conv import baseline2d
from functools import partial
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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
        print(f"Using fixed one-hot pad lengths: receptor={pad_rec_len}, ligand={pad_lig_len}\n")

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

    return recs, ligs, labels

#######################################__Model__##################################################

def setup_model(train_loader, device, run, residue):
    # 2. Setup device, model, optimizer, and loss function
    if residue:
        rec_emb_sample, _,  _ = next(iter(train_loader))
        embed_dim = rec_emb_sample.size(2)  # Assuming the embedding dimension is the third dimension
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

    # Binary Cross Entropy Loss for binary classification
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()  # more stable than BCELoss with separate sigmoid

    # Give model to WandB
    wandb.watch(model, log="all", log_freq=100)

    return model, optimizer, criterion


def train(device, run, model, optimizer, criterion, train_loader, val_loader):
    # 3. Training loop
    for i in range(run.config.epochs):

        model.train()
        total_loss = 0.0
        for rec_emb, lig_emb, labels in train_loader:                
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

def test(device, model, test_loader):
    # 5. Test the model
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for rec_emb, lig_emb, labels in test_loader:
            rec_emb, lig_emb, labels = rec_emb.to(device), lig_emb.to(device), labels.to(device)

            labels = labels.float()  # Convert labels to float for BCELoss

            logits = model(rec_emb, lig_emb)

            # Accuracy
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Collect all preds and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Test accuracy: {correct/total*100:.2f}%\n")
    print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds, labels=[0,1])}\n")

    return all_preds, all_labels

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test ML Architecture for Protein Interaction Prediction")
    parser.add_argument('--token', type=str, default=None, required=True, help='Token for WandB login')
    parser.add_argument('--path', type=str, default=None, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--random_forest', action='store_true', help='If set, run Random Forest classifier instead of NN')
    parser.add_argument('--lrp', action='store_true', help='If set, perform LRP analysis after testing')
    parser.add_argument('--judith_test', action='store_true', help='If set, use Judith gold standard test set')
    parser.add_argument('--save_predictions', action='store_true', help='If set, save predictions to a CSV file')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--residue', action='store_true', help='Use residue-level embeddings instead of mean embeddings')
    group.add_argument('--one_hot', action='store_true', help='Use one-hot encoding instead of embeddings')



    args = parser.parse_args()

    # global
    global AA_ALPHABET, aa_to_idx

    # Define the amino acid alphabet (can adapt if you want gaps, X, etc.)
    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXBU"
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    
    # set embedding type
    if args.residue:
        embedding_type = "residue"
    else:
        embedding_type = "mean"
    
    # embedding cache dir
    cache_dir = f"/nfs/scratch/pinder/negative_dataset/my_repository/embeddings/sequence/ESM3/{embedding_type}"
    
    # print encoding type
    if not args.one_hot:
        print(f"Using {embedding_type} embeddings\n")
        print(f"Using embedding cache dir: {cache_dir}\n")
    else:
        print("Using one-hot encoding\n")
    
    # print dataset path
    print(f"Using dataset path: {args.path}")

    # dataset paths to csv files
    train_csv = os.path.join(args.path, "train.csv")
    val_csv = os.path.join(args.path, "val.csv")
    test_csv = os.path.join(args.path, "test.csv")

    # set up judith test set if specified
    if args.judith_test:
        test_csv = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/judith_gold_standard/test_pinder.csv"
        print(f"Using Judith test set at: {test_csv}")

    # !!!!arguments not sure what will happen with in the future!!!!
    kernel_size = 2  # Default kernel size, can be adjusted
    wandb_mode = "disabled"  # Change to "online" to enable WandB logging, "disabled" to disable it

    # execute random forest if specified
    if args.random_forest:
        handle_random_forest(train_csv, test_csv)
        sys.exit(0)

    # PIPELINE
    device, run, train_loader, val_loader, test_loader = setup(cache_dir, train_csv, val_csv, test_csv, args.residue, args.one_hot, kernel_size, wandb_mode)
    model, optimizer, criterion = setup_model(train_loader, device, run, args.residue)
    train(device, run, model, optimizer, criterion, train_loader, val_loader)
    all_preds, all_labels = test(device, model, test_loader)

    if args.save_predictions:
        print("Saving test predictions to CSV file...\n")
        test_df = pd.read_csv(test_csv)
        test_df['pred'] = all_preds
        test_df['check_labels'] = all_labels
        out_csv = os.path.join(args.path, "test_predictions.csv")

    # LRP analysis if specified
    if args.lrp:
        print("Executing LRP on test set!!\n")
        exec_lrp_simple_interaction_net(model, device, test_loader, args.one_hot)

if __name__ == "__main__":
    main()


