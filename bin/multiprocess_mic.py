import os
import json
import math
import pandas as pd
import numpy as np
from minepy import MINE
import concurrent.futures
from multiprocessing import cpu_count

  
def split_data(data, label_names, test_fraction=0.2, random_state=42):
    """Split input data into training and testing sets
    
    data        <pd.DataFrame> : Expects dataframe with genes as columns and samples as rows
    label_names <array>        : List of name(s) that are labels
    """
    
    test = data.sample(
        frac=test_fraction, 
        random_state=random_state,
        axis=0)
    train = data.loc[~data.index.isin(test.index)]
    
    X_test = test[test.columns.difference(label_names)]
    y_test = test[label_names]
    
    X_train = train[train.columns.difference(label_names)]
    y_train = train[label_names]
    
    feature_labels = [X_test.columns.tolist(), y_test.columns.tolist()]
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), feature_labels

def run_mic(args):

    _id, arg_dict = args[0], args[1]
    data = arg_dict["data"]
    print("\tRunning {0}...".format(str(_id)))
    
    X_train, X_test, y_train, y_test, feature_labels = split_data(data, [_id], test_fraction=0.3)
    results = np.zeros(X_train.shape[1])

    for i in range(X_train.shape[1]):
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(X_train[:, i], y_train[:, 0])
        results[i] = mine.mic()

    output = pd.DataFrame(results, index=feature_labels[0], columns=feature_labels[1])
    output.to_csv(os.path.join("..", "mic_output", _id + ".tsv"), sep="\t")
    print(_id + " complete.")
    
def run_pools(
    func,
    arg_iter,
    arg_dict):
    
    cores = arg_dict['workers']
    pools = int(math.ceil(len(arg_iter) / arg_dict['workers']))
    if pools < 1:
        pools = 1
    print("Processing {0} pool(s) on {1} core(s)...".format(pools, cores))
    
    it_list = []
    range_number = 0
    for x in range(pools):
        it_list.append([iter for iter in arg_iter[range_number:range_number + arg_dict['workers']]])
        range_number += arg_dict['workers']
    print("Divided data across {0} pool(s).\n".format(pools))
    
    batch_number = 1
    for batch in it_list:
        print("Starting: {0}...".format(str([x[0] for x in batch])))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=arg_dict['workers']) as executor:
            for gene in zip(batch, executor.map(func, batch)):
                print("{0} has been processed.".format(gene[0][0]))
        
        print('Processing of batch {0} of {1} complete...\n'.format(batch_number, pools))
        batch_number += 1
        

if __name__ == '__main__':
    ### Load data and sample metadata
    data_path = os.path.join(
        os.getcwd(),
        "..",
        "data",
        "S_cereviseae_compendia_recount_bio"
    )

    print("Loading data...")
    with open(
            os.path.join(data_path, 'aggregated_metadata.json'), 'r') as jsonfile:
        metadata_file = json.load(jsonfile)

    tables = {}
    for k, v in metadata_file['experiments'].items():
        tables[v["accession_code"]] = v

    metadata = pd.DataFrame(tables).T

    data = pd.read_csv(os.path.join(data_path, "SACCHAROMYCES_CEREVISIAE.tsv"), sep="\t", header=0, index_col=0).T
    print("Loaded data with dimensions:", str(data.shape))

    ### Load gene metadata
    # Get metadata for genes and extract genes with "transporter" annotation
    print("Loading metadata...")
    
    gene_mapper = pd.read_csv(os.path.join(
        os.getcwd(),
        "..",
        "data",
        "yeast_orf_dict.csv"
    ), header=None, names=["id", "symbol", "name", "description"])

    transporters = gene_mapper.loc[gene_mapper["description"].str.contains("transporter")]
    transporters_list = transporters["id"].tolist()

    gene_map = {}
    for i, r in gene_mapper.iterrows():
        _id = str(r["id"])
        _name = str(r["symbol"])
        if _name != "nan":
            gene_map[_id] = _name
        else:
            gene_map[_id] = _id
        
    print("Processing data...")
    arg_dict = {
        "workers": cpu_count(),
        "data": data
    }

    transporter_list = transporters["id"].tolist()
    arg_iter = [[gene, arg_dict] for gene in transporter_list]
    run_pools(
        run_mic,
        arg_iter,
        arg_dict)