import torch
import torch.functional as F

def forward(self, rec_hid, lig_hid):
    mat = torch.matmul(rec_hid, lig_hid.transpose(1, 2))
    # Shape: (B, L, L) where B is batch size, L is length of receptor and ligand sequences (Can be different)

    # Add dimensions for pooling
    C = mat.unsqueeze(0)  # [1,b,n,m]
    C = C.permute(1, 0, 2, 3) # [b,1,n,m]

    # extract batch size
    B, _, _, _ = C.shape


    # Shape: (B,  1, L, L) where B is batch size, 1 is for single channel, and L is length of sequences

    # Lokales Max-Pooling über nicht-überlappende Regionen
    # Zero-Padding so, dass n,m durch pool_size teilbar werden
    pool_size = 2
    pad_h = (pool_size - C.size(2) % pool_size) % pool_size
    pad_w = (pool_size - C.size(3) % pool_size) % pool_size
    C_padded = F.pad(C, (0, pad_w, 0, pad_h), mode='constant', value=0)
    # C_padded: [1,1,⌈n/ps⌉,⌈m/ps⌉] where ps is pool_size
    C_max = F.max_pool2d(C_padded,
                            kernel_size=pool_size,
                            stride=pool_size)    # Form: [1,1,⌈n/ps⌉,⌈m/ps⌉]
    
    # Globales Pooling: Mittelwert und Varianz
    flat = C_max.view(B, -1)
    mu = flat.mean(dim=1, keepdim=True)                   # [B,1]
    sigma = flat.var(dim=1, unbiased=False, keepdim=True) # [B,1]

    # test with gamma = 0.5 and eta = 1.0
    gamma = 0.5
    eta = 1.0

    # Sparsifizierung: nur Werte > μ + γ·σ behalten
    threshold = mu + gamma * sigma
    M = F.relu(flat - threshold)     # alle <= threshold werden 0

    # average over only the positive entries per sample
    mask      = M > 0                                      # [B,N]
    sum_pos   = M.sum(dim=1, keepdim=True)                 # [B,1]
    count_pos = mask.sum(dim=1, keepdim=True).clamp(min=1) # [B,1]
    y_bar     = sum_pos / count_pos                        # [B,1]

    # 6) Finale „extremisierende“ Aktivierung (Logistic)
    y = torch.sigmoid(eta * (y_bar - 0.5))

    # y ist ein Skalar in [0,1] mit size [B]
    return y.view(B)