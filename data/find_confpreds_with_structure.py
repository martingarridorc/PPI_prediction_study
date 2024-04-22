import pandas as pd
import ast
import pickle

def main(model, dataset):
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
    df_new.columns = ['prot1', 'prot2', 'pdb_matches']
    #df_new.to_csv(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/conf_ppis_with_structures_{dataset}_{model}.csv', sep=',', index=False)
    print(f'Found {len(df_new)} common pairs')
    df_new['pdb_matches'] = df_new['pdb_matches'].apply(ast.literal_eval)
    confpred_dict = {frozenset([row['prot1'], row['prot2']]): row['pdb_matches'][0] if row['pdb_matches'] else None for _, row in df_new.iterrows()}

    with open(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/confpred_dict_{dataset}_{model}.pkl', 'wb') as f:
        pickle.dump(confpred_dict, f)

if __name__ == "__main__":
    main(dataset='Intra2', model='crossattention')        