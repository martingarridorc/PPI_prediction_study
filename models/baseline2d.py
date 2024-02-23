import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.data as d


class baseline2d(nn.Module):
    def __init__(self, embed_dim, h = 0):
        super(baseline2d, self).__init__()

        if h == 0:
            h = int(embed_dim//4)
        h2 = int(h//4)  #new    
        h3 = int(h2//4) #new

        self.conv = nn.Conv2d(h3, 1, kernel_size=(2, 2))
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.maxpool = nn.MaxPool2d(kernel_size=4)  #new

        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)
        #self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.Linear(h, h2)
        #self.bn2 = nn.BatchNorm1d(h2)

        self.fc3 = nn.Linear(h2, h3)
        #self.bn3 = nn.BatchNorm1d(h3)
        #self.fc4 = nn.Linear(h3, 1)    

        self.sigmoid = nn.Sigmoid()


    def forward(self, protein1, protein2):

        x1 = protein1.to(torch.float32)
        x2 = protein2.to(torch.float32)

        x1 = self.ReLU(self.fc1(x1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))
        
        x2 = self.ReLU(self.fc1(x2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))



        mat = torch.einsum('ik,jk->ijk', x1, x2)    # normale matrix multiplikation
        mat = mat.permute(2, 0, 1)
        
        mat = self.conv(mat.unsqueeze(0))

        x = self.pool(mat)    
                   
        m = torch.max(x)

        pred = self.sigmoid(m)
        pred = pred[None]

        return pred, mat



    def shifted_sigmoid(x, balance_point):
        x = balance_point + torch.sigmoid((x - balance_point) * 10)
        return x
    

    def batch_iterate(self, batch, device, layer, emb_dir):
            pred = []
            for i in range(len(batch['interaction'])):
                id1 = batch['name1'][i]
                id2 = batch['name2'][i]
                seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).to(device)
                seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).to(device)
                p, cm = self.forward(seq1, seq2)
                pred.append(p)
            return torch.stack(pred)  

