import torch
import torch.nn as nn


# 1. Define a simple feed-forward classifier
class SimpleInteractionNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim=1536):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim / 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim / 2, 2)  # Binary classification (interaction vs no interaction)
        )

    def forward(self, rec_emb, lig_emb):
        # e.g. shape of rec_emb and lig_emb: torch.Size([B, 1536])
        x = torch.cat([rec_emb, lig_emb], dim=1)  # Concatenate embeddings along the embedding dimension
        # e.g. x shape: torch.Size([B, 3072])

        return self.fc(x) # shape: (B, 2)