import torch
import torch.nn as nn


# 1. Define a simple feed-forward classifier
class SimpleInteractionNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, rec_emb, lig_emb):
        # e.g. rec_emb shape: torch.Size([2, 596, 1536]), lig_emb shape: torch.Size([2, 131, 1536])
        # x = torch.cat([rec_emb, lig_emb], dim=1)  # Concatenate embeddings along the sequence length dimension
        # e.g. x shape: torch.Size([2, 727, 1536])

        # e.g. shape of rec_emb and lig_emb: torch.Size([2, 1536])
        x = torch.cat([rec_emb, lig_emb], dim=1)  # Concatenate embeddings along the embedding dimension
        # e.g. x shape: torch.Size([2, 3072])

        return self.fc(x) # shape: (B, 2)