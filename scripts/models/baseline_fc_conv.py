import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Define a simple feed-forward classifier
class disabled_baseline2d(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, kernel_size=2):
        self.residue = True

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

        mat = mat.permute(0, 3, 1, 2) # [B, H, L, L], "Channel" is hidden dimension and must be at 1

        mat = self.conv(mat) # Apply convolution

        mat = self.pool(mat) # Apply max pooling

        max = torch.amax(mat, dim=(2, 3)) # Max over the pooled output
        # max is a scalar in [0,1] with size [B, 1]

        logit = max.squeeze(-1) # Flatten to 1D tensor
        # logit is a scalar in [0,1] with size [B]

        return logit
    

class disabled_baseline2d(nn.Module):
    def __init__(self, embed_dim, h3 = 64, kernel_size = 2, pooling = 'avg'):
        self.residue = True

        super(baseline2d, self).__init__()

        if embed_dim < 30:  #usually just the case for one-hot-encoding / might need to check this differently
            h = h3
            h2 = h3
            h3 = h3
        else:
            h = int(embed_dim//4)
            h2 = int(h//4)
            h3 = h3

        self.conv = nn.Conv2d(h3, 1, kernel_size=kernel_size, padding='same')
        if pooling == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)  

        self.ReLU = nn.ReLU()
        self.fc1 = nn.utils.parametrizations.spectral_norm(nn.Linear(embed_dim, h))
        #self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.utils.parametrizations.spectral_norm(nn.Linear(h, h2))
        #self.bn2 = nn.BatchNorm1d(h2)

        self.fc3 = nn.utils.parametrizations.spectral_norm(nn.Linear(h2, h3))
        #self.bn3 = nn.BatchNorm1d(h3)
        #self.fc4 = nn.Linear(h3, 1)

        self.sigmoid = nn.Sigmoid()

        self.cross12 = nn.MultiheadAttention(h3, 4, dropout=0.1, batch_first=True)
        self.cross21 = nn.MultiheadAttention(h3, 4, dropout=0.1, batch_first=True)
        self.ln12 = nn.LayerNorm(h3)
        self.ln21 = nn.LayerNorm(h3)


    def forward(self, x1 = None, x2 = None, x1_kpm = None, x2_kpm = None):

        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        x1 = self.ReLU(self.fc1(x1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))
        
        x2 = self.ReLU(self.fc1(x2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))

        attn12, _ = self.cross12(query=x1, key=x2, value=x2, key_padding_mask=x2_kpm)
        x1 = self.ln12(x1 + attn12)

        attn21, _ = self.cross21(query=x2, key=x1, value=x1, key_padding_mask=x1_kpm)
        x2 = self.ln21(x2 + attn21)

        mat = torch.einsum('bih,bjh->bijh', x1, x2)    # normale matrix multiplikation

        # valid positions are ~pad_mask
        valid_r = (~x1_kpm).unsqueeze(2)   # (B, Lr, 1)
        valid_l = (~x2_kpm).unsqueeze(1)   # (B, 1, Ll)
        valid_pairs = (valid_r & valid_l)       # (B, Lr, Ll)

        # mat is (B, Lr, Ll, h3) at this point
        mat = mat.masked_fill(~valid_pairs.unsqueeze(-1), 0.0)  # or -inf if you want strict max later

        mat = mat.permute(0, 3, 1, 2)
        
        mat = self.conv(mat)

        x = self.pool(mat)
                   
        max = torch.amax(x, dim=(2, 3)) # Max over the pooled output

        logit = max.squeeze(-1)

        return logit


class baseline2d(nn.Module):
    def __init__(self, embed_dim, h3 = 64, kernel_size = 2, pooling = 'avg'):
        self.residue = True

        super(baseline2d, self).__init__()

        if embed_dim < 30:  #usually just the case for one-hot-encoding / might need to check this differently
            h = h3
            h2 = h3
            h3 = h3
        else:
            h = int(embed_dim//4)
            h2 = int(h//4)
            h3 = h3

        self.conv = nn.Conv2d(h3, 1, kernel_size=kernel_size, padding='same')
        if pooling == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)  

        self.ReLU = nn.ReLU()
        self.fc1 = nn.utils.parametrizations.spectral_norm(nn.Linear(embed_dim, h))
        self.fc2 = nn.utils.parametrizations.spectral_norm(nn.Linear(h, h2))
        self.fc3 = nn.utils.parametrizations.spectral_norm(nn.Linear(h2, h3))

        self.sigmoid = nn.Sigmoid()

        self.layernorm = nn.LayerNorm(h3)




    def forward(self, x1 = None, x2 = None, x1_kpm = None, x2_kpm = None):

        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        x1 = self.ReLU(self.fc1(x1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))

        x2 = self.ReLU(self.fc1(x2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))

        x1 = self.layernorm(x1)
        x2 = self.layernorm(x2)

        # scale before outer product
        d = x1.size(-1)
        x1 = x1 / math.sqrt(d)
        x2 = x2 / math.sqrt(d)

        # valid positions are ~pad_mask
        valid_r = (~x1_kpm).unsqueeze(2)   # (B, Lr, 1)
        valid_l = (~x2_kpm).unsqueeze(1)   # (B, 1, Ll)
        valid_pairs = (valid_r & valid_l)       # (B, Lr, Ll)

        mat = torch.einsum('bih,bjh->bijh', x1, x2)    # normale matrix multiplikation
        mat = mat.masked_fill(~valid_pairs.unsqueeze(-1), -torch.inf)
        mat = mat.permute(0, 3, 1, 2)

        mat = self.conv(mat)
        x = self.pool(mat)

        # valid_pairs: (B, Lr, Ll) bool
        mask2d = valid_pairs.unsqueeze(1).float()         # (B, 1, Lr, Ll)
        mask2d_p = self.pool(mask2d) > 0.0                # (B, 1, Lr', Ll') bool

        logit = masked_logmeanexp(
            x, mask2d_p,
            dim=(2, 3),
            tau=1.0,
            keepdim=False
        )  # -> (B, 1)

        logit = logit.squeeze(1)

        return logit

def masked_logsumexp(x, mask, dim, tau=1.0, keepdim=False):
    """
    x:   tensor
    mask: bool tensor broadcastable to x, True = valid, False = invalid/pad
    dim: int or tuple of ints to reduce
    tau: temperature (smaller => closer to max)
    """
    # avoid all-masked -> -inf -> nan later
    # (we'll clamp count to >=1 in the return)
    x = x / tau
    x = x.masked_fill(~mask, float("-inf"))
    out = torch.logsumexp(x, dim=dim, keepdim=keepdim) * tau
    return out

def masked_logmeanexp(x, mask, dim, tau=1.0, keepdim=False, eps=1e-8):
    x = x / tau
    x = x.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(x, dim=dim, keepdim=keepdim)
    # N = number of valid entries reduced over
    N = mask.float().sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    out = (lse - torch.log(N + eps)) * tau
    return out
