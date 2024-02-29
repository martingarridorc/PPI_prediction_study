import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import models.dscript_like as dscript_like
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
                forward_expansion=1, poolsize=50, pre_cnn_drop=0.5):
        super(ICAN_cross, self).__init__()

        self.transformer_block = TransformerBlock(embed_dim, num_heads, transformer_drop, forward_expansion)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be a divisible by the number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, value, key, query, mask):
        batch = query.shape[0]

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        values = V.reshape(batch, value.shape[1], self.num_heads, self.head_dim)
        keys = K.reshape(batch, key.shape[1], self.num_heads, self.head_dim)
        queries = Q.reshape(batch, query.shape[1], self.num_heads, self.head_dim)


        energy = torch.einsum("bqhs,bkhs->bhqk", [queries,keys]) / self.embed_dim ** 0.5

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("bhqv,bvhs->bqhs", [attention, values]).reshape(batch, query.shape[1], self.num_heads * self.head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)    # use nn.MultiheadAttention instead
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.fnn = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion* embed_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.fnn(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    

class CrossAttInteraction(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2, forward_expansion=1):
        super(CrossAttInteraction, self).__init__()
    
        self.transformer_block = TransformerBlock(embed_dim, num_heads, dropout, forward_expansion)
        
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

        cross1 = self.transformer_block(x1, x1, x2, mask1)
        cross2 = self.transformer_block(x2, x2, x1, mask2)

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
    def __init__(self, embed_dim, num_heads, dropout=0.2, forward_expansion=1):
        super(SelfAttInteraction,self).__init__()

        self.transformer_block = TransformerBlock(embed_dim, num_heads, dropout, forward_expansion)
        

        h = int(embed_dim//4)
        h2 = int(h//4)    
        h3 = int(h2//4)

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

    def forward(self, protein1, protein2, mask1, mask2):
        x1 = protein1.to(torch.float32)
        x2 = protein2.to(torch.float32)

        self1 = self.transformer_block(x1, x1, x1, mask1)
        self2 = self.transformer_block(x2, x2, x2, mask2)

        x1 = self.ReLU(self.fc1(self1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))
    
        x2 = self.ReLU(self.fc1(self2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))

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
                    p, cm = self.forward(seq1, seq2, None, None)
                    pred.append(p)
                return torch.stack(pred)


class AttentionDscript(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.25, forward_expansion=1):
        super(AttentionDscript, self).__init__()
        #self.transformer_block = TransformerBlock(embed_dim, num_heads, dropout, forward_expansion)
        self.multihead = MultiHeadAttention(embed_dim, num_heads)
        self.pymultihead = nn.MultiheadAttention(embed_dim, num_heads)
        self.dscript = dscript_like.DScriptLike(embed_dim)

    def forward(self, protein1, protein2, mask1, mask2):
        x1 = protein1.to(torch.float32)
        x2 = protein2.to(torch.float32)

        self1, self1weights = self.pymultihead(x1, x1, x1, mask1)
        self2, self2weights = self.pymultihead(x2, x2, x2, mask2)

        return self.dscript(self1, self2)
    
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


class AttentionRichoux(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, forward_expansion=1):
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

        self1, self1weights = self.attention(x1, x1, x1, mask1)
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
    


