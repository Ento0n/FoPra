import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Define a simple feed-forward classifier
class baseline2d(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, kernel_size=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.conv = nn.Conv2d(
            in_channels=hidden_dim, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.pool = nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, rec_emb, lig_emb):
        rec_hid = self.fc(rec_emb)
        lig_hid = self.fc(lig_emb)
        # Shape: (B, L, H) where B is batch size, L is length, and H is hidden dimension

        mat = torch.einsum("bih,bjh->bijh", rec_hid, lig_hid) # Outer product over h dimensions
        # Shape: (B, L, L, H) where B is batch size, L is length of receptor and ligand sequences (Can be different) and H is hidden dimension

        mat = torch.tanh(mat)  # bring values to [-1, 1] values are crazy high otherwise

        mat = mat.permute(0, 3, 1, 2) # [B, H, L, L], "Channel" is hidden dimension and must be at 1

        mat = self.conv(mat) # Apply convolution

        mat = self.pool(mat) # Apply max pooling

        max = torch.amax(mat, dim=(2, 3)) # Max over the pooled output
        # max is a scalar in [0,1] with size [B, 1]

        logit = torch.sigmoid(max)
        logit = logit.view(-1) # Flatten to 1D tensor
        # logit is a scalar in [0,1] with size [B]

        return logit
