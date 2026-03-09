import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
import torch.nn.functional as F
import math

class TUnA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, hid_dim = 64, dropout=0.25, ff_dim=256, rffs=1028, cross=True):
        super(TUnA, self).__init__()

        self.residue = True

        self.cross = cross
        self.hid_dim = hid_dim
        self.num_heads = num_heads

        self.Cross_Intra = CrossEncoderLayer(hid_dim, num_heads, ff_dim, dropout)
        self.Intra = EncoderLayer(hid_dim, num_heads, ff_dim, dropout)
    
        self.Cross_Inter = CrossEncoderLayer(hid_dim, num_heads, ff_dim, dropout)
        self.Inter = EncoderLayer(hid_dim, num_heads, ff_dim, dropout)

        self.lin1 = spectral_norm(nn.Linear(embed_dim, hid_dim))

        self.pred_layer = VanillaRFFLayer(hid_dim, rffs, 1, likelihood="binary_logistic")

    def forward(self, protein1, protein2, mask1=None, mask2=None):
        # mask1/mask2 are key padding masks (B, L), True=pad
        x1 = self.lin1(protein1)
        x2 = self.lin1(protein2)

        # build masks
        m1_self = self._kpm_to_self_mask(mask1)
        m2_self = self._kpm_to_self_mask(mask2)

        if self.cross:
            m1_cross = self._kpm_to_cross_mask(mask2, x1.size(1))  # keys = x2
            m2_cross = self._kpm_to_cross_mask(mask1, x2.size(1))  # keys = x1
            x1_encoded = self.Cross_Intra(x1, x2, m1_cross)
            x2_encoded = self.Cross_Intra(x2, x1, m2_cross)
        else:
            x1_encoded = self.Intra(x1, m1_self)
            x2_encoded = self.Intra(x2, m2_self)

        x12 = torch.cat((x1_encoded, x2_encoded), dim=1)
        x21 = torch.cat((x2_encoded, x1_encoded), dim=1)

        x12_mask = self.combine_masks(m1_self, m2_self)
        x21_mask = self.combine_masks(m2_self, m1_self)

        if self.cross:
            x12_encoded = self.Cross_Inter(x12, x21, x21_mask)
            x21_encoded = self.Cross_Inter(x21, x12, x21_mask)
        else:
            x12_encoded = self.Inter(x12, x12_mask)
            x21_encoded = self.Inter(x21, x21_mask)

        x12_mask_2d = x12_mask[:,0,:,0]
        x21_mask_2d = x21_mask[:,0,:,0]

        x12_interact = torch.sum(x12_encoded*x12_mask_2d[:,:,None], dim=1)/x12_mask_2d.sum(dim=1, keepdims=True)
        x21_interact = torch.sum(x21_encoded*x21_mask_2d[:,:,None], dim=1)/x21_mask_2d.sum(dim=1, keepdims=True)

        ppi_feature_vector, _ = torch.max(torch.stack([x12_interact, x21_interact], dim=-1), dim=-1)

        predictions = self.pred_layer(ppi_feature_vector)

        # return torch.sigmoid(predictions)

        return predictions.squeeze(-1)
  

    def create_mask(self, tensor: torch.Tensor):
        mask = (tensor == 0).all(dim=-1)
        return mask

    def create_square_mask(self, x):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
        N, seq_len, _ = x.size()  # batch size and sequence length
        mask = torch.zeros((N, seq_len, seq_len), device=device)

        for i in range(N):
            # Find the length of the sequence (excluding padding)
            lens = (x[i].sum(dim=-1) != 0).sum().item()

            # Create a square mask for the non-padded sequence
            mask[i, :lens, :lens] = 1

        # Expand the mask to 4D: [batch, 1, seq_len, seq_len]
        mask = mask.unsqueeze(1)
        return mask

    def create_cross_mask(self, tensorA, tensorB):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
        lenA, lenB = tensorA.size(1), tensorB.size(1)
        combined_mask = torch.zeros(tensorA.size(0), 1, lenA, lenB, device=device)

        # Create a 1D mask for tensorA
        maskA = (tensorA.sum(dim=-1) != 0)
        combined_mask[:, :, :, :lenB] = maskA.unsqueeze(1).unsqueeze(3)

        # Create a 1D mask for tensorB
        maskB = (tensorB.sum(dim=-1) != 0)
        combined_mask[:, :, :lenA, :] = maskB.unsqueeze(1).unsqueeze(2)

        return combined_mask

    def combine_masks(self, maskA, maskB):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
        lenA, lenB = maskA.size(2), maskB.size(2)
        combined_mask = torch.zeros(maskA.size(0), 1, lenA + lenB, lenA + lenB, device=device)
        combined_mask[:, :, :lenA, :lenA] = maskA
        combined_mask[:, :, lenA:, lenA:] = maskB
        return combined_mask

    def _kpm_to_square_mask(self, kpm: torch.Tensor) -> torch.Tensor:
        # kpm: (B, L) where True = pad. Return (B, 1, L, L) with 1 for valid positions.
        if kpm is None:
            return None
        if kpm.dim() == 1:
            kpm = kpm.unsqueeze(0)  # (1, L)
        valid = ~kpm  # True = valid
        # outer product to get (B, L, L) valid pairs
        square = valid.unsqueeze(2) & valid.unsqueeze(1)
        return square.unsqueeze(1).float()

    def _kpm_to_self_mask(self, kpm: torch.Tensor) -> torch.Tensor:
        # kpm: (B, L) with True = pad
        if kpm is None:
            return None
        if kpm.dim() == 1:
            kpm = kpm.unsqueeze(0)
        valid = ~kpm  # True = valid
        square = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, L, L)
        return square.unsqueeze(1).float()  # (B, 1, L, L)

    def _kpm_to_cross_mask(self, kpm_keys: torch.Tensor, q_len: int) -> torch.Tensor:
        # key padding mask for cross-attn: only mask keys
        # kpm_keys: (B, Lk) with True = pad
        if kpm_keys is None:
            return None
        if kpm_keys.dim() == 1:
            kpm_keys = kpm_keys.unsqueeze(0)
        valid = ~kpm_keys  # (B, Lk)
        # (B, 1, Lq, Lk), broadcast over heads
        return valid[:, None, None, :].expand(-1, 1, q_len, -1).float()
    
    def old_code(self):
        # Old code using torch encoder
        #self.Intra_Encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, batch_first=True)
        #self.Intra_Encoder = nn.TransformerEncoder(self.Intra_Encoder_layer, num_layers=num_layers)
        #self.Inter_Encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, batch_first=True)
        #self.Inter_Encoder = nn.TransformerEncoder(self.Inter_Encoder_layer, num_layers=num_layers)
        '''
            mask1 = self.create_mask(x1)
            mask2 = self.create_mask(x2)

            #reduce dim to hid_dim (part of intra-encoder)
            x1 = self.lin1(x1)
            x2 = self.lin1(x2)

            #first encoder
            x1_encoded = self.Intra_Encoder(x1, src_key_padding_mask=mask1)
            x2_encoded = self.Intra_Encoder(x2, src_key_padding_mask=mask2)

            #combine both permutations, proteins ...
            x12 = torch.cat((x1_encoded, x2_encoded), dim=1)
            x21 = torch.cat((x2_encoded, x1_encoded), dim=1)

            #... and masks
            x12_mask = torch.cat((mask1, mask2), dim=1)
            x21_mask = torch.cat((mask2, mask1), dim=1)

            #second encoder
            x12_encoded = self.Inter_Encoder(x12, src_key_padding_mask=x12_mask)
            x21_encoded = self.Inter_Encoder(x21, src_key_padding_mask=x21_mask)

            # average over the real parts of the sequence
            x12_interact = torch.sum(x12_encoded*x12_mask[:,:,None], dim=1)/x12_mask.sum(dim=1, keepdims=True)
            x21_interact = torch.sum(x21_encoded*x21_mask[:,:,None], dim=1)/x21_mask.sum(dim=1, keepdims=True)
            '''
        # old code not using spectral norm (for test)
        # no specnorm self.Intra2 = EncoderLayer_nospecnorm(hid_dim, num_heads, ff_dim, dropout)
        # no specnorm self.Inter2 = EncoderLayer_nospecnorm(hid_dim, num_heads, ff_dim, dropout)
        # no specnorm self.lin2 = nn.Linear(embed_dim, hid_dim)
        '''    
            mask1 = self.create_square_mask(x1)
            mask2 = self.create_square_mask(x2)

            x1 = self.lin2(x1)
            x2 = self.lin2(x2)

            x1_encoded = self.Intra2(x1, mask1)
            x2_encoded = self.Intra2(x2, mask2)

            x12 = torch.cat((x1_encoded, x2_encoded), dim=1)
            x21 = torch.cat((x2_encoded, x1_encoded), dim=1)

            x12_mask = self.combine_masks(mask1, mask2)
            x21_mask = self.combine_masks(mask2, mask1)

            x12_encoded = self.Inter2(x12, x12_mask)
            x21_encoded = self.Inter2(x21, x21_mask)

            x12_mask_2d = x12_mask[:,0,:,0]
            x21_mask_2d = x21_mask[:,0,:,0]

            x12_interact = torch.sum(x12_encoded*x12_mask_2d[:,:,None], dim=1)/x12_mask_2d.sum(dim=1, keepdims=True)
            x21_interact = torch.sum(x21_encoded*x21_mask_2d[:,:,None], dim=1)/x21_mask_2d.sum(dim=1, keepdims=True)
            '''
        return None
    

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, ff_dim, dropout, activation_fn='swish'):
        super().__init__()
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)
        
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        
        self.sa = Attention(hid_dim, n_heads, dropout)
        self.ff = Feedforward(hid_dim, ff_dim, dropout, activation_fn)
        
    def forward(self, trg, mask=None):

        #trg_1 = trg
        #trg = self.sa(trg, trg, trg, trg_mask)
        #trg = self.ln1(trg_1 + self.do1(trg))
        #
        #trg = self.ln2(trg + self.do2(self.ff(trg)))

        trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, mask)))
        trg = self.ln2(trg + self.do2(self.ff(trg)))


        return trg

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
    

# from https://github.com/Wang-lab-UCSD/uncertaintyAwareDeepLearn/blob/main/uncertaintyAwareDeepLearn/classic_rffs.py
_ACCEPTED_LIKELIHOODS = ("gaussian", "binary_logistic", "multiclass")
class VanillaRFFLayer(nn.Module):
    """
    A PyTorch layer for random features-based regression, binary classification and
    multiclass classification.

    Args:
        in_features: The dimensionality of each input datapoint. Each input
            tensor should be a 2d tensor of size (N, in_features).
        RFFs: The number of RFFs generated. Must be an even number. The larger RFFs,
            the more accurate the approximation of the kernel, but also the greater
            the computational expense. We suggest 1024 as a reasonable value.
        out_targets: The number of output targets to predict. For regression and
            binary classification, this must be 1. For multiclass classification,
            this should be the number of possible categories in the data.
        gp_cov_momentum (float): A "discount factor" used to update a moving average
            for the updates to the covariance matrix. 0.999 is a reasonable default
            if the number of steps per epoch is large, otherwise you may want to
            experiment with smaller values. If you set this to < 0 (e.g. to -1),
            the precision matrix will be generated in a single epoch without
            any momentum.
        gp_ridge_penalty (float): The initial diagonal value for computing the
            covariance matrix; useful for numerical stability so should not be
            set to zero. 1e-3 is a reasonable default although in some cases
            experimenting with different settings may improve performance.
        likelihood (str): One of "gaussian", "binary_logistic", "multiclass".
            Determines how the precision matrix is calculated. Use "gaussian"
            for regression.
        amplitude (float): The kernel amplitude. This is the inverse of
            the lengthscale. Performance is not generally
            very sensitive to the selected value for this hyperparameter,
            although it may affect calibration. Defaults to 1.
        random_seed: The random seed for generating the random features weight
            matrix. IMPORTANT -- always set this for reproducibility. Defaults to
            123.

    Shape:
        - Input: :math:`(N, H_{in})` where :math:`N` means number of datapoints.
          Only 2d input arrays are accepted.
        - Output: :math:`(N, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out}` = out_targets.

    Examples::

        >>> m = nn.VanillaRFFLayer(20, 1)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 1])
    """

    def __init__(self, in_features: int, RFFs: int, out_targets: int=1,
            gp_cov_momentum = 0.999, gp_ridge_penalty = 1e-3,
            likelihood = "gaussian", amplitude = 1.,
            random_seed: int=123, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if not isinstance(out_targets, int) or not isinstance(RFFs, int) or \
                not isinstance(in_features, int):
            raise ValueError("out_targets, RFFs and in_features must be integers.")
        if out_targets < 1 or RFFs < 1 or in_features < 1:
            raise ValueError("out_targets, RFFs and in_features must be > 0.")
        if RFFs <= 1 or RFFs % 2 != 0:
            raise ValueError("RFFs must be an even number greater than 1.")
        if likelihood not in _ACCEPTED_LIKELIHOODS:
            raise ValueError(f"Likelihood must be one of {_ACCEPTED_LIKELIHOODS}.")
        if likelihood in ["gaussian", "binary_logistic"] and out_targets != 1:
            raise ValueError("For regression and binary_logistic likelihoods, "
                    "only one out target is expected.")
        if likelihood == "multiclass" and out_targets <= 1:
            raise ValueError("For multiclass likelihood, more than one out target "
                    "is expected.")

        self.in_features = in_features
        self.out_targets = out_targets
        self.fitted = False
        self.momentum = gp_cov_momentum
        self.ridge_penalty = gp_ridge_penalty
        self.RFFs = RFFs
        self.likelihood = likelihood
        self.amplitude = amplitude
        self.random_seed = random_seed
        self.num_freqs = int(0.5 * RFFs)
        self.feature_scale = math.sqrt(2. / float(self.num_freqs))

        self.register_buffer("weight_mat", torch.zeros((in_features, self.num_freqs), **factory_kwargs))
        self.output_weights = nn.Parameter(torch.empty((RFFs, out_targets), **factory_kwargs))
        self.register_buffer("covariance", torch.zeros((RFFs, RFFs), **factory_kwargs))
        self.register_buffer("precision", torch.zeros((RFFs, RFFs), **factory_kwargs))
        self.reset_parameters()


    def train(self, mode=True) -> None:
        """Sets the layer to train or eval mode when called
        by the parent model. NOTE: Setting the model to
        eval if it was previously in train will cause
        the covariance matrix to be calculated. This can
        (if the number of RFFs is large) be an expensive calculation,
        so expect model.eval() to take a moment in such cases."""
        if mode:
            self.fitted = False
        else:
            if not self.fitted:
                self.covariance[...] = torch.linalg.pinv(self.ridge_penalty *
                    torch.eye(self.precision.size()[0], device = self.precision.device) +
                    self.precision)
            self.fitted = True


    def reset_parameters(self) -> None:
        """Set parameters to initial values. We don't need to use kaiming
        normal -- in fact, that would set the variance on our sqexp kernel
        to something other than 1 (which is ok, but might be unexpected for
        the user)."""
        self.fitted = False
        with torch.no_grad():
            rgen = torch.Generator()
            rgen.manual_seed(self.random_seed)
            self.weight_mat = torch.randn(generator = rgen,
                    size = self.weight_mat.size())
            self.output_weights[:] = torch.randn(generator = rgen,
                    size = self.output_weights.size())
            self.covariance[:] = (1 / self.ridge_penalty) * torch.eye(self.RFFs)
            self.precision[:] = 0.


    def reset_covariance(self) -> None:
        """Resets the covariance to the initial values. Useful if
        planning to generate the precision & covariance matrices
        on the final epoch."""
        self.fitted = False
        with torch.no_grad():
            self.precision[:] = 0.
            self.covariance[:] = (1 / self.ridge_penalty) * torch.eye(self.RFFs)

    def forward(self, input_tensor: torch.Tensor, update_precision: bool = False,
            get_var: bool = False) -> torch.Tensor:
        """Forward pass. Only updates the precision matrix if update_precision is
        set to True.

        Args:
            input_tensor (Tensor): The input x values. Must be a 2d tensor.
            update_precision (bool): If True, update the precision matrix. Only
                do this during training.
            get_var (bool): If True, obtain the variance on the predictions. Only
                do this when generating model predictions (not necessary during
                training).

        Returns:
            logits (Tensor): The output predictions, of size (input_tensor.shape[0],
                    out_targets)
            var (Tensor): Only returned if get_var is True. Indicates variance on
                predictions.

        Raises:
            RuntimeError: A RuntimeError is raised if get_var is set to True
                but model.eval() has never been called."""
        if len(input_tensor.size()) != 2:
            raise ValueError("Only 2d input tensors are accepted by "
                    "VanillaRFFLayer.")
        rff_mat = self.amplitude * input_tensor @ self.weight_mat
        rff_mat = self.feature_scale * torch.cat([torch.cos(rff_mat), torch.sin(rff_mat)], dim=1)
        logits = rff_mat @ self.output_weights

        if update_precision:
            self.fitted = False
            self._update_precision(rff_mat, logits)

        if get_var:
            if not self.fitted:
                raise RuntimeError("Must call model.eval() to generate "
                        "the covariance matrix before requesting a "
                        "variance calculation.")
            with torch.no_grad():
                var = self.ridge_penalty * (self.covariance @ rff_mat.T).T
                var = torch.sum(rff_mat * var, dim=1)
            return logits, var
        return logits
    
    def _update_precision(self, rff_mat: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Updates the precision matrix. If momentum is < 0, the precision
        matrix is updated using a sum over all minibatches in the epoch;
        this calculation therefore needs to be run only once, on the
        last epoch. If momentum is > 0, the precision matrix is updated
        using the momentum term selected by the user. Note that for multi-class
        classification, we actually compute an upper bound; see Liu et al. 2022.;
        since computing the full Hessian would be too expensive if there is
        a large number of classes."""
        with torch.no_grad():
            if self.likelihood == 'binary_logistic':
                prob = torch.sigmoid(logits)
                prob_multiplier = prob * (1. - prob)
            elif self.likelihood == 'multiclass':
                prob = torch.max(torch.softmax(logits), dim=1)
                prob_multiplier = prob * (1. - prob)
            else:
                prob_multiplier = 1.

            gp_feature_adjusted = torch.sqrt(prob_multiplier) * rff_mat
            precision_matrix_minibatch = gp_feature_adjusted.T @ gp_feature_adjusted
            if self.momentum < 0:
                self.precision += precision_matrix_minibatch
            else:
                self.precision[...] = (
                    self.momentum * self.precision
                    + (1 - self.momentum) * precision_matrix_minibatch)