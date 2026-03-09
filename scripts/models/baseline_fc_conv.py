import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np


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

        super().__init__()

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


class disabled_baseline2d(nn.Module):
    def __init__(self, embed_dim, h3 = 64, kernel_size = 2, pooling = 'avg'):
        self.residue = True

        super().__init__()

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

################################### Timos Cross attention Model #######################################

class baseline2d(nn.Module):
    def __init__(self, embed_dim, num_heads=4, h3=64, dropout=0.2, ff_dim=256, pooling='avg', kernel_size=2):
        super().__init__()

        self.residue = True

        if embed_dim < 30:  #usually just the case for one-hot-encoding / might need to check this differently
            h = h3
            h2 = h3
            h3 = h3
        else:
            h = int(embed_dim//4)
            h2 = int(h//4)

        self.cross_encoder = CrossEncoderLayer(h3, num_heads, ff_dim, dropout)

        self.conv = nn.Conv2d(h3, 1, kernel_size=kernel_size, padding='same')
        if pooling == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)
        self.fc2 = nn.Linear(h, h2)
        self.fc3 = nn.Linear(h2, h3)

        self.sigmoid = nn.Sigmoid()


    def forward(self, protein1, protein2, mask1=None, mask2=None):
        x1 = protein1.to(torch.float32)
        x2 = protein2.to(torch.float32)

        if x1.dim() == 2:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0)

        # masks come in as key padding masks (True = pad). Convert to valid-mask
        # with shape broadcastable to attention energy (B, H, Lq, Lk).
        if mask1 is not None:
            if mask1.dim() == 1:
                mask1 = mask1.unsqueeze(0)
            mask1 = (~mask1).unsqueeze(1).unsqueeze(2)
        if mask2 is not None:
            if mask2.dim() == 1:
                mask2 = mask2.unsqueeze(0)
            mask2 = (~mask2).unsqueeze(1).unsqueeze(2)

        x1 = self.ReLU(self.fc1(x1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))
    
        x2 = self.ReLU(self.fc1(x2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))

        # use key padding mask for the cross sequence (keys)
        x1 = self.cross_encoder(x1, x2, mask2)
        x2 = self.cross_encoder(x2, x1, mask1)

        mat = torch.einsum('bik,bjk->bijk', x1, x2)    # outer product over last dimension
        mat = mat.permute(0, 3, 1, 2)
        mat = self.conv(mat)
        x = self.pool(mat)    
           
        # reduce over spatial dims only, keep batch
        m = torch.amax(x, dim=(2, 3))

        # pred = self.sigmoid(m)
        # pred = pred[None]

        logit = m.squeeze(1)

        return logit


    def batch_iterate(self, batch, device, layer, emb_dir, embedding=True):
            pred = []
            for i in range(len(batch['interaction'])):
                id1 = batch['name1'][i]
                id2 = batch['name2'][i]
                if embedding:
                    seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).to(device)
                    seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).to(device)
                else:
                    seq1 = batch['sequence_a'][i]
                    seq2 = batch['sequence_b'][i]
                    seq1 = d.sequence_to_vector(seq1)
                    seq2 = d.sequence_to_vector(seq2)
                    seq1 = torch.tensor(np.array(seq1)).to(device)
                    seq2 = torch.tensor(np.array(seq2)).to(device)
                p, cm = self.forward(seq1, seq2)
                pred.append(p)
            return torch.stack(pred) 


# modified EncoderLayer for cross attention
class CrossEncoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, ff_dim, dropout, activation_fn='swish'):
        super().__init__()
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)
        
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        
        self.sa = Attention(hid_dim, n_heads, dropout)
        self.ff = Feedforward(hid_dim, ff_dim, dropout, activation_fn)
        
    def forward(self, trg, cross, mask=None):

        trg = self.ln1(trg + self.do1(self.sa(trg, cross, cross, mask)))
        trg = self.ln2(trg + self.do2(self.ff(trg)))

        return trg   
    

#from https://github.com/Wang-lab-UCSD/TUnA/blob/main/results/bernett/TUnA/model.py, essentially a copy 
# of 'attention is all you need' (as is nn.MultiHeadAttention), but with spectral normalization
class Attention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"

        # Linear transformations for query, key, and value
        self.w_q = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_k = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_v = spectral_norm(nn.Linear(hid_dim, hid_dim))

        # Final linear transformation
        self.fc = spectral_norm(nn.Linear(hid_dim, hid_dim))

        # Dropout for attention
        self.do = nn.Dropout(dropout)

        # Scaling factor for the dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # Compute query, key, value matrices [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Reshape for multi-head attention and permute to bring heads forward
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Compute attention weights [batch size, n heads, sent len_Q, sent len_K]
        attention = self.do(F.softmax(energy, dim=-1))
        
        # Apply attention to the value matrix
        x = torch.matmul(attention, V)  # transpose

        # Reshape and concatenate heads
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # Final linear transformation [batch size, sent len_Q, hid dim]
        x = self.fc(x)

        return x


class Feedforward(nn.Module):
    def __init__(self, hid_dim, ff_dim, dropout, activation_fn):
        super().__init__()

        self.hid_dim = hid_dim
        self.ff_dim = ff_dim

        self.fc_1 = spectral_norm(nn.Linear(hid_dim, ff_dim))  
        self.fc_2 = spectral_norm(nn.Linear(ff_dim, hid_dim))  

        self.do = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(activation_fn)
    
    def _get_activation_fn(self, activation_fn):
        """Return the corresponding activation function."""
        if activation_fn == "relu":
            return nn.ReLU()
        elif activation_fn == "gelu":
            return nn.GELU()
        elif activation_fn == "elu":
            return nn.ELU()
        elif activation_fn == "swish":
            return nn.SiLU()
        elif activation_fn == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_fn == "mish":
            return nn.Mish()
        # Add other activation functions if needed
        else:
            raise ValueError(f"Activation function {activation_fn} not supported.")
    
    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = self.do(self.activation(self.fc_1(x)))
        # x = [batch size, ff dim, sent len]

        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]
        return x
