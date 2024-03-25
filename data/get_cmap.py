import Bio.PDB
import numpy as np
import plotly.graph_objects as go
from pcmap import contactMap
import pandas as pd

pdb_code = "2K7W"
pdb_filename = "data/pdb2k7w.ent"

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float64)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

# with biopython
structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
model = structure[0]
dist_matrix = calc_dist_matrix(model["A"], model["B"])
contact_map = dist_matrix < 12.0
fig = go.Figure(go.Heatmap(z=contact_map.astype(np.float64),
                           x=[f"{str(res.resname)}:{str(res.id[1])}" for res in model["B"]],
                           y=[f"{str(res.resname)}:{str(res.id[1])}" for res in model["A"]],
                           colorscale='gray',
                           reversescale=True))
fig.update_layout(
    width=1000,
    height=1000)
fig.write_html('data/heatmap.html')

fig = go.Figure(go.Heatmap(z=dist_matrix.astype(np.float64),
                           x=[f"{str(res.resname)}:{str(res.id[1])}" for res in model["B"]],
                           y=[f"{str(res.resname)}:{str(res.id[1])}" for res in model["A"]],
                           colorscale='blues',
                           reversescale=True))
fig.update_layout(
    width=1000,
    height=1000)
fig.write_html('data/heatmap_dist.html')


# with cmap
c = contactMap(pdb_filename, dist=8)['data']
root_ids = {f"{item['root']['chainID']}:{item['root']['resID']}" for item in c}
partner_ids = {f"{item['partners'][i]['chainID']}:{item['partners'][i]['resID']}" for item in c for i in range(len(item['partners']))}
ids = list(root_ids.union(partner_ids))
# sort ids: first all IDs that start with A, then all IDs that start with B.
# after A/B comes a : and then a number. Sort by the number.
ids = sorted(ids, key=lambda x: (x[0], int(x.split(':')[1])))
cmap = pd.DataFrame(np.zeros((len(ids), len(ids)), np.float64))
cmap.index = ids
cmap.columns = ids
for i in range(len(c)):
    row_id = c[i]['root']['chainID'] + ":" + c[i]['root']['resID']
    for j in range(len(c[i]['partners'])):
        row_id2 = c[i]['partners'][j]['chainID'] + ":" + c[i]['partners'][j]['resID']
        cmap.at[row_id, row_id2] = 1
        cmap.at[row_id2, row_id] = 1

# filter such that index only starts with 'A' and columns only start with 'B'
cmap = cmap[cmap.index.str.startswith('A')]
cmap = cmap[cmap.columns[cmap.columns.str.startswith('B')]]

fig = go.Figure(go.Heatmap(z=cmap.values,
                            x=cmap.columns,
                            y=cmap.index,
                           colorscale='gray',
                           reversescale=True))
fig.update_layout(
    width=1000,
    height=1000)
fig.write_html('data/heatmap2.html')