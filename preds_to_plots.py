import data.find_confpreds_with_structure as fcws
import data.get_cmap as gc
import data.data as d
import models.dscript_like as dscript_like
import models.baseline2d as baseline2d
import models.fc2_20_2_dense as richoux
import models.attention as attention
import numpy as np
import ast
import os
import requests
import pickle
import torch
import matplotlib.pyplot as plt

def main(dataset='Intra1', model='TUnA'):
    confpred_dict = load_dict(dataset, model)
    for interaction in list(confpred_dict.keys())[5:]:
        list_interaction = list(interaction)
        id1 = list_interaction[0]
        id2 = list_interaction[1]
        complex_id = confpred_dict[interaction]

        cmap = get_pred_cmap(id1, id2, complex_id, model=model, emb_name='esm2_t33_650', emb_type='per_tok')
        plot_predcmap(cmap, f'{complex_id}_{model}')
        pdb_file = download_pdb_files([complex_id])
        real_cmap = gc.get_cmap(pdb_filename=pdb_file)
        gc.plot_cmap(real_cmap)
        dist_map, x_labels, y_labels = gc.get_distmap(complex_id, pdb_file)
        gc.plot_distmap(dist_map, x_labels, y_labels)
        gc.plot_distmap(cmap, None, None)

def load_dict(dataset, model):
    with open(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/confpred_dict_{dataset}_{model}.pkl', 'rb') as f:
        return pickle.load(f)
    

def get_pred_cmap(id1, id2, complex_id, model, emb_name='esm2_t33_650', emb_type='per_tok'):
    model_path = f'/nfs/home/students/t.reim/bachelor/pytorchtest/models/pretrained/{model}_{emb_name}.pt'
    layer = int(emb_name.split('t')[1].split('_')[0])
    emb_dir = "/nfs/scratch/t.reim/embeddings/" + emb_name + "/" + emb_type + "/"

    #add way to load different models
    model = get_model(model, emb_sizes[emb_name])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).unsqueeze(0)
    seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).unsqueeze(0)
    model.eval()
    pred, cmap = model(seq1, seq2)
    print(f'{complex_id}: {pred}')
    cmap = cmap.squeeze().detach().numpy() 
    return cmap

def plot_predcmap(cmap, name):
    plt.imshow(cmap, cmap='hot')
    plt.colorbar()
    plt.savefig(f'/nfs/home/students/t.reim/bachelor/pytorchtest/plots/contact_map_{name}.png')
    plt.clf()

def download_pdb_files(complex_ids, directory='/nfs/home/students/t.reim/bachelor/pytorchtest/data/pdb_files'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for pdb_id in complex_ids:
        file_path = os.path.join(directory, f'{pdb_id}.pdb')

        if not os.path.exists(file_path):
            url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
            response = requests.get(url)

            if response.status_code == 200:
                with open(file_path, 'w') as file:
                    file.write(response.text)
            else:
                print(f'Failed to download PDB file for {pdb_id}')
    
    return file_path            

emb_sizes = {
    'esm2_t48_3B': 5120,
    'esm2_t36_650M': 2560,
    'esm2_t33_650': 1280
}

def get_model(model, insize, h3=64, num_heads=8, dropout=0.2, ff_dim=256, pooling='avg', kernel_size=2):
    model_mapping = {
            "dscript_like": dscript_like.DScriptLike(embed_dim=insize, d=100, w=7, h=50, x0=0.5, k=20),
            "richoux": richoux.FC2_20_2Dense(embed_dim=insize),
            "baseline2d": baseline2d.baseline2d(embed_dim=insize, h3=h3),
            "selfattention": attention.SelfAttInteraction(embed_dim=insize, num_heads=num_heads, dropout=dropout),
            "crossattention": attention.CrossAttInteraction(embed_dim=insize, num_heads=num_heads, h3=h3, dropout=dropout,
                                                            ff_dim=ff_dim, pooling=pooling, kernel_size=kernel_size),
            "ICAN_cross": attention.ICAN_cross(embed_dim=insize, num_heads=num_heads, cnn_drop=dropout),
            "AttDscript": attention.AttentionDscript(embed_dim=insize, num_heads=num_heads, dropout=dropout),
            "Rich-ATT": attention.AttentionRichoux(embed_dim=insize, num_heads=num_heads, dropout=dropout),
            "TUnA": attention.TUnA(embed_dim=insize, num_heads=num_heads, dropout=dropout)
        }
    return model_mapping[model]

main(dataset='Intra1', model='crossattention')