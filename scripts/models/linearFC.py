import torch
import torch.nn as nn


# 1. Define a simple feed-forward classifier
class disabled_LinearFC(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(input_dim, 1536)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.utils.parametrizations.spectral_norm(nn.Linear(1536, 768)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.utils.parametrizations.spectral_norm(nn.Linear(768, 1)),  # Output single value for binary classification
            # nn.Sigmoid()
        )

    def forward(self, rec_emb, lig_emb):
        # e.g. shape of rec_emb and lig_emb: torch.Size([B, 1536])
        x = torch.cat([rec_emb, lig_emb], dim=1)  # Concatenate embeddings along the embedding dimension
        # e.g. x shape: torch.Size([B, 3072])

        return self.fc(x).squeeze(-1) # shape: (B,)
    

# 1. Define a simple feed-forward classifier
class LinearFC(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super().__init__()

        self.residue = False

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),  # Output single value for binary classification
        )

    def forward(self, rec_emb, lig_emb):
        # e.g. shape of rec_emb and lig_emb: torch.Size([B, 1536])
        x = torch.cat([rec_emb, lig_emb], dim=1)  # Concatenate embeddings along the embedding dimension
        # e.g. x shape: torch.Size([B, 3072])

        return self.fc(x).squeeze(-1) # shape: (B,)