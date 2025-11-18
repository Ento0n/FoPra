import torch
import torch.nn as nn


# 1. Define a simple feed-forward classifier
class LinearFC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1536),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1536, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(768, 1),  # Output single value for binary classification
            # nn.Sigmoid()
        )

    def forward(self, rec_emb, lig_emb):
        # e.g. shape of rec_emb and lig_emb: torch.Size([B, 1536])
        x = torch.cat([rec_emb, lig_emb], dim=1)  # Concatenate embeddings along the embedding dimension
        # e.g. x shape: torch.Size([B, 3072])

        return self.fc(x).squeeze(-1) # shape: (B,)