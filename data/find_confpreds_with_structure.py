import pandas as pd

model = 'TUnA'
dataset = 'Intra1'
mapp = {
    'Intra0': 'val',
    'Intra1': 'train',
    'Intra2': 'test',
}
sett = mapp[dataset]
# Read the CSV files
df1 = pd.read_csv(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/confident_{sett}_pred_{model}.csv', sep='\t')
df2 = pd.read_csv(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/ppis_with_structures_{dataset}_pos_rr.csv', sep=',',skiprows=1)


pairs1 = [frozenset((row[0], row[1])) for row in df1.values]
pairs2 = [frozenset((row[0], row[1])) for row in df2.values]

# Find the intersection of the two sets
common_pairs = set(pairs1) & set(pairs2)


df_new = df2[df2.apply(lambda row: frozenset((row[0], row[1])) in common_pairs, axis=1)]
df_new.to_csv(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/conf_ppis_with_structures_{dataset}_{model}.csv', sep=',', index=False)


'''
# test
pairs1 = {frozenset(x) for x in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]}
pairs2 = {frozenset(x) for x in [(10, 9), (11, 12), (13, 14), (15, 16), (17, 18)]}

# Find the intersection of the two sets
common_pairs = pairs1 & pairs2

[print(tuple(pair)) for pair in common_pairs]
'''