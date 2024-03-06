import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import models.dscript_like as dscript_like
import torch.nn.utils.spectral_norm as spectral_norm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.data as d


class ICAN_CNN(nn.Module):
    def __init__(self, embed_dim, poolsize, dropout):
        super(ICAN_CNN, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=5, stride=1, padding = 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding = 2)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(poolsize)  # Add adaptive pooling layer
        self.dense_1 = nn.Linear(32 * poolsize, 128)
        self.dense_2 = nn.Linear(128, 32)
        self.dense_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid_func = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        output = torch.transpose(input, -1, -2)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = self.conv2(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = self.adaptive_pool(output)  # Apply adaptive pooling
        output = output.view(-1, output.size(1) * output.size(2))
        
        #fully connected layer
        output = self.dense_1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense_2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense_3(output)
        output = self.sigmoid_func(output)
       
        return output


class ICAN_cross(nn.Module):
    def __init__(self, embed_dim, num_heads, cnn_drop = 0.4, transformer_drop=0.25,
                 poolsize=50, pre_cnn_drop=0.5):
        super(ICAN_cross, self).__init__()

        self.torchencoderlayer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.torchencoder = nn.TransformerEncoder(self.torchencoderlayer, num_layers=3)

        self.CNN = ICAN_CNN(embed_dim, poolsize, cnn_drop)
        self.pre_cnn_drop = nn.Dropout(pre_cnn_drop)

    def forward(self, protein1, protein2, mask1, mask2):
        x1 = protein1.to(torch.float32)
        x2 = protein2.to(torch.float32)

        cross1 = self.transformer_block(x1, x1, x2, mask1)
        cross2 = self.transformer_block(x2, x2, x1, mask2)   

        cross1_drop = self.pre_cnn_drop(cross1)
        cross2_drop = self.pre_cnn_drop(cross2)

        out1 = self.CNN(cross1_drop)
        out2 = self.CNN(cross2_drop)


        return max(out1, out2).view(1)
    
    def batch_iterate(self, batch, device, layer, emb_dir):
            pred = []
            for i in range(len(batch['interaction'])):
                id1 = batch['name1'][i]
                id2 = batch['name2'][i]
                seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).unsqueeze(0).to(device)
                seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).unsqueeze(0).to(device)
                p = self.forward(seq1, seq2, None, None)
                pred.append(p)
            return torch.stack(pred)


class CrossAttInteraction(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(CrossAttInteraction, self).__init__()
    

        self.torchencoderlayer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.torchencoder = nn.TransformerEncoder(self.torchencoderlayer, num_layers=3)

        self.multihead = nn.MultiheadAttention(embed_dim, num_heads)
        
        h = int(embed_dim//4)
        h2 = int(h//4)    
        h3 = int(h2//4) 

        self.conv = nn.Conv2d(h3, 1, kernel_size=(2, 2))
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.maxpool = nn.MaxPool2d(kernel_size=4)

        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)
        self.fc2 = nn.Linear(h, h2)
        self.fc3 = nn.Linear(h2, h3)

        self.sigmoid = nn.Sigmoid()


    def forward(self, protein1, protein2, mask1, mask2):
        x1 = protein1.to(torch.float32)
        x2 = protein2.to(torch.float32)

        cross1, cross1weights = self.multihead(x1, x1, x2, mask1)
        cross2, cross2weights = self.multihead(x2, x2, x1, mask2)

        # option 1, copy of 2dbaseline, using both cross1 and cross2
        x1 = self.ReLU(self.fc1(cross1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))
    
        x2 = self.ReLU(self.fc1(cross2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))

        mat = torch.einsum('bik,bjk->bijk', x1, x2)    # normale matrix multiplikation
        mat = mat.permute(0, 3, 1, 2)
        mat = self.conv(mat)
        x = self.pool(mat)    
           
        m = torch.max(x)

        pred = self.sigmoid(m)
        pred = pred[None]


        #option 2, to be added: using either cross1 or cross2

        #x1 = self.ReLU(self.fc1(cross1))
        #x1 = self.ReLU(self.fc2(x1))
        #x1 = self.ReLU(self.fc3(x1))


        return pred, mat
    

    def batch_iterate(self, batch, device, layer, emb_dir):
            pred = []
            for i in range(len(batch['interaction'])):
                id1 = batch['name1'][i]
                id2 = batch['name2'][i]
                seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).unsqueeze(0).to(device)
                seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).unsqueeze(0).to(device)
                p, cm = self.forward(seq1, seq2, None, None)
                pred.append(p)
            return torch.stack(pred)


class SelfAttInteraction(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2, encoder_layers=1):
        super(SelfAttInteraction,self).__init__()

        #self.transformer_block = TransformerBlock(embed_dim, num_heads, dropout, forward_expansion)
        h = int(embed_dim//4)
        h2 = int(h//4)    
        h3 = 64

        self.torchencoderlayer = nn.TransformerEncoderLayer(d_model=h3, nhead=num_heads)
        self.torchencoder = nn.TransformerEncoder(self.torchencoderlayer, num_layers=encoder_layers)

        self.multihead = nn.MultiheadAttention(h3, num_heads)


        self.conv = nn.Conv2d(h3, 1, kernel_size=(2, 2))
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.maxpool = nn.MaxPool2d(kernel_size=4)

        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)
        #self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.Linear(h, h2)
        #self.bn2 = nn.BatchNorm1d(h2)

        self.fc3 = nn.Linear(h2, h3)
        #self.bn3 = nn.BatchNorm1d(h3)
        #self.fc4 = nn.Linear(h3, 1) 

        self.sigmoid = nn.Sigmoid()

    def forward(self, protein1, protein2, mask1, mask2, use_encoder=True):
        x1 = protein1.to(torch.float32)
        x2 = protein2.to(torch.float32)


        x1 = self.ReLU(self.fc1(x1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))
    
        x2 = self.ReLU(self.fc1(x2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))

        if use_encoder:
            x1 = self.torchencoder(x1)
            x2 = self.torchencoder(x2)
        else:
            x1, _ = self.multihead(x1, x1, x1, mask1)
            x2, _ = self.multihead(x2, x2, x2, mask2)

        mat = torch.einsum('bik,bjk->bijk', x1, x2)    # normale matrix multiplikation
        mat = mat.permute(0, 3, 1, 2)
        mat = self.conv(mat)
        x = self.pool(mat)    
           
        m = torch.max(x)

        pred = self.sigmoid(m)
        pred = pred[None]


        return pred, mat


    def batch_iterate(self, batch, device, layer, emb_dir):
                pred = []
                for i in range(len(batch['interaction'])):
                    id1 = batch['name1'][i]
                    id2 = batch['name2'][i]
                    seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).unsqueeze(0).to(device)
                    seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).unsqueeze(0).to(device)
                    p, _ = self.forward(seq1, seq2, None, None, True)
                    pred.append(p)
                return torch.stack(pred)


class AttentionDscript(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.25):
        super(AttentionDscript, self).__init__()


        self.torchencoderlayer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.torchencoder = nn.TransformerEncoder(self.torchencoderlayer, num_layers=3)

        self.pymultihead = nn.MultiheadAttention(embed_dim, num_heads)

        self.dscript = dscript_like.DScriptLike(embed_dim)

    def forward(self, protein1, protein2, mask1, mask2):
        x1 = protein1.to(torch.float32)
        x2 = protein2.to(torch.float32)

        #self1, _ = self.pymultihead(x1, x1, x1, mask1)
        #self2, _ = self.pymultihead(x2, x2, x2, mask2)

        self1 = self.torchencoder(x1)
        self2 = self.torchencoder(x2)

        return self.dscript(self1, self2)
    
    def batch_iterate(self, batch, device, layer, emb_dir):
                pred = []
                for i in range(len(batch['interaction'])):
                    id1 = batch['name1'][i]
                    id2 = batch['name2'][i]
                    seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).unsqueeze(0).to(device)
                    seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).unsqueeze(0).to(device)
                    p, _ = self.forward(seq1, seq2, None, None)
                    pred.append(p)
                return torch.stack(pred)


class TUnA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, hid_dim = 64, dropout=0.25):
        super(TUnA, self).__init__()

        self.hid_dim = hid_dim
        self.num_heads = num_heads

        self.Intra_Encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, batch_first=True)
        self.Intra_Encoder = nn.TransformerEncoder(self.Intra_Encoder_layer, num_layers=num_layers)

        self.Intra2 = EncoderLayer(hid_dim, num_heads, 256, dropout)

        self.Inter_Encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, batch_first=True)
        self.Inter_Encoder = nn.TransformerEncoder(self.Inter_Encoder_layer, num_layers=num_layers)

        self.Inter2 = EncoderLayer(hid_dim, num_heads, 256, dropout)

        self.lin1 = spectral_norm(nn.Linear(embed_dim, hid_dim))

        self.pred_layer = VanillaRFFLayer(hid_dim, 1024, 1)

    def forward(self, proteins, spec_norm = True):
        #split protein and create masks
        x1, x2 = proteins.split(1, dim=1)
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)

        if(spec_norm):
            mask1 = self.create_square_mask(x1)
            mask2 = self.create_square_mask(x2)

            x1 = self.lin1(x1)
            x2 = self.lin1(x2)

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
        else:
        
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


        ppi_feature_vector, _ = torch.max(torch.stack([x12_interact, x21_interact], dim=-1), dim=-1)

        predictions = self.pred_layer(ppi_feature_vector)

        return torch.sigmoid(predictions)
  

    def create_mask(self, tensor: torch.Tensor):
        mask = (tensor == 0).all(dim=-1)
        return mask

    def create_square_mask(self, x):
        if torch.cuda.is_available():
            device = torch.device("cuda")
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

    def combine_masks(self, maskA, maskB):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        lenA, lenB = maskA.size(2), maskB.size(2)
        combined_mask = torch.zeros(maskA.size(0), 1, lenA + lenB, lenA + lenB, device=device)
        combined_mask[:, :, :lenA, :lenA] = maskA
        combined_mask[:, :, lenA:, lenA:] = maskB
        return combined_mask    

# mean embeddings after attetion are meaningless
class AttentionRichoux(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super(AttentionRichoux, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads,dropout=dropout)

        #from richoux
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, 20)
        self.fc2 = nn.Linear(20, 20)

        self.fc3 = nn.Linear(embed_dim, 20)
        self.fc4 = nn.Linear(20, 20)

        self.fc5 = nn.Linear(40, 20)
        self.fc6 = nn.Linear(20, 1)
        self.classes = (0,1)

    def forward(self, x1, x2, mask1, mask2):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        self1, _ = self.attention(x1, x1, x1, mask1)
        self2, self2weights = self.attention(x2, x2, x2, mask2)

        x1 = torch.mean(self1, dim=1)
        x2 = torch.mean(self2, dim=1)

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))

        x2 = F.relu(self.fc3(x2))
        x2 = F.relu(self.fc4(x2))

        x = torch.cat((x1,x2), 1)

        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        x = torch.sigmoid(x)

        return x.squeeze(0)
    
    def batch_iterate(self, batch, device, layer, emb_dir):
                pred = []
                for i in range(len(batch['interaction'])):
                    id1 = batch['name1'][i]
                    id2 = batch['name2'][i]
                    seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).unsqueeze(0).to(device)
                    seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).unsqueeze(0).to(device)
                    p = self.forward(seq1, seq2, None, None)
                    pred.append(p)
                return torch.stack(pred)


# can be ignored, may be deleted
class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]
    


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


class Attention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        if torch.cuda.is_available():
            device = torch.device("cuda")

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
        x = torch.matmul(attention, V)

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