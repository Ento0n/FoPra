import torch
import torch.nn as nn

class FCEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.encoder(x)
    
class RichouxInteractionNet(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.rec_encoder = FCEncoder(embed_dim)
        self.lig_encoder = FCEncoder(embed_dim)

        self.classifier  = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary classification (interaction vs no interaction)
        )

    def forward(self, rec_emb, lig_emb):
        # e.g. shape of rec_emb and lig_emb: torch.Size([B, 1536])
        rec_hid = self.rec_encoder(rec_emb)  # e.g. rec_hid shape: torch.Size([B, 512])
        lig_hid = self.lig_encoder(lig_emb)  # e.g. lig_hid shape: torch.Size([B, 512])
        x = torch.cat([rec_hid, lig_hid], dim=1)  # Concatenate embeddings along the embedding dimension
        # e.g. x shape: torch.Size([B, 1024])

        return self.classifier(x) # shape: (B, 2)