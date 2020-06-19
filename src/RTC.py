import os
import glob
import sys
import gc

import torch
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from text_line_dataset import TextLineInMemoryDataset
import metrics
from utils import mkdir, files_exist
import datetime

def voting(A):
    A  += np.finfo(np.float).eps
    A = A/(A+A.T)
    for i in range(A.shape[0]):
        A[i,i] = np.finfo(np.float).eps
    T = (A>0.5).sum(axis=1)
    return T.argsort()[::-1]

def eval_model(model, data, out_folder=None):
    #--- build data
    samples = len(data.pairs)
    feats_len = data.pairs[0]['x'].shape[0]
    features = np.zeros((samples, feats_len))
    targets = np.zeros(samples)
    for i,sample in enumerate(data.pairs):
        features[i,:] = sample['x'].numpy()
        targets[i] = int(sample['t'])
    
    pred = model.predict(features)
    T = {}
    for i,sample in enumerate(data.pairs):
        p = sample['parent']
        if p not in T:
            ne = len(data.order[p])
            T[p] = {'Prob': np.zeros((ne,ne)),
                    'order':data.order[p]}
        idxi = T[p]['order'].index(sample["z"][0])
        idxj = T[p]['order'].index(sample["z"][1])
        T[p]['Prob'][idxi,idxj] = pred[i]

    Fst = {}
    Ktd = {}
    for page in T.keys():
        s = data.order[page]
        t = voting(T[page]['Prob'])
        Fst[page] = metrics.spearman_footrule_distance(
            list(range(len(s))), list(t)
        )
        Ktd[page] = {"val":0, "normalized":0}
        (Ktd[page]['val'],
        Ktd[page]['normalized']) = metrics.kendall_tau_distance(
                list(range(len(s))), 
                list(t),
                both=True)
        # --- save results
        if out_folder:
            with open(os.path.join(out_folder, page + ".pickle"), "wb") as fh:
                pickle.dump((t, T[page]), fh)
    return (Fst, Ktd)


def main():
    """
    """
    n_epochs = 3
    out_folder = sys.argv[1]
    tr_data = sys.argv[2]
    tr_processed_data = os.path.join(tr_data, "processed")
    val_data = sys.argv[3]
    val_processed_data = os.path.join(val_data, "processed")
    te_data = sys.argv[4]
    te_processed_data = os.path.join(te_data, "processed")
    categories = sys.argv[5].split()
    n_estimators = int(sys.argv[6])
    max_depth= int(sys.argv[7])
    exp_id = sys.argv[8]
    out_folder = os.path.join(out_folder, exp_id)
    mkdir(out_folder)
    n_jobs = len(os.sched_getaffinity(0))
    random_state = datetime.datetime.now().microsecond

    #--- use "test" set_id to force pair generation
    tr_dataset = TextLineInMemoryDataset(
        tr_data,
        set_id="test",
        processed_data=tr_processed_data,
        categories=categories,
        transform=None,
        force_regenerate=False,
    )
    feats_len = tr_dataset.pairs[0]['x'].shape[0]
    tr_samples = len(tr_dataset.pairs)
    tr_features = np.zeros((tr_samples, feats_len))
    tr_targets = np.zeros(tr_samples)
    for i,sample in enumerate(tr_dataset.pairs):
        tr_features[i,:] = sample['x'].numpy()
        tr_targets[i] = int(sample['t'])


    rfc = RandomForestClassifier(random_state=random_state, 
            n_jobs=n_jobs, 
            n_estimators=n_estimators, 
            max_depth=max_depth)

    print("Training Random Forest Classifier....")
    rfc.fit(tr_features,tr_targets)
    #--- save model
    with open(out_folder + "/RFC_model.pickle",'wb') as fh:
        pickle.dump(rfc, fh, protocol=pickle.HIGHEST_PROTOCOL)
    #--- clean memory 
    del tr_features
    del tr_targets
    del tr_dataset
    gc.collect()
    print("Trainning done.")
    val_dataset = TextLineInMemoryDataset(
        val_data,
        set_id="val",
        processed_data=val_processed_data,
        categories=categories,
        transform=None,
        force_regenerate=False,
    )

    val_metrics = eval_model(rfc, val_dataset, out_folder=None)
    Fst = val_metrics[0]
    Ktd = val_metrics[1]
    print("Val Avg F(s,t): {}".format(sum(Fst.values()) / len(Fst)))
    kv = 0
    kn = 0
    for i,v in Ktd.items():
        kv += v['val']
        kn += v['normalized']
    print("Val Avg K(s,t): {}".format(kv/len(Fst)))
    print("Val Avg norm K(s,t): {}".format(kn/len(Fst)))
    #--- clean memory 
    del val_dataset
    gc.collect()

    te_dataset = TextLineInMemoryDataset(
        te_data,
        set_id="test",
        processed_data=te_processed_data,
        categories=categories,
        transform=None,
        force_regenerate=False,
    )
    te_folder = os.path.join(out_folder, 'test')
    mkdir(te_folder)
    te_metrics = eval_model(rfc, te_dataset, out_folder=te_folder)
    Fst = te_metrics[0]
    Ktd = te_metrics[1]
    print("Test Avg F(s,t): {}".format(sum(Fst.values()) / len(Fst)))
    kv = 0
    kn = 0
    for i,v in Ktd.items():
        kv += v['val']
        kn += v['normalized']
    print("Test Avg K(s,t): {}".format(kv/len(Fst)))
    print("Test Avg norm K(s,t): {}".format(kn/len(Fst)))


if __name__ == "__main__":
    main()
