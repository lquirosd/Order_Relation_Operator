import numpy as np
import sys
import pickle

import metrics
import torch


def basic_ro(centers):
    """ 
    sort the elements in the left-rigth-top-dowm order given its center
    """
    return np.lexsort((centers[:,0],centers[:,1]))
    

def main():
    
    out_folder = sys.argv[1]
    gt_file = sys.argv[2]
    cat_size = int(sys.argv[3])
    #with open(gt_file, 'rb') as fh:
    #    data = pickle.load(fh)
    
    gt_data, gt_relatives, gt_order = torch.load(gt_file)
    #print(data)
    
    id_to_idx = {}
    for idx, d in enumerate(gt_data):
        id_to_idx[d["parent"] + d["id"]] = idx

    Fst = {}
    Ktd = {}

    for img_name in gt_order.keys():
        z = gt_order[img_name]
        points = np.zeros((len(z), 2))
        for j, element in enumerate(z):
            points[j] = gt_data[id_to_idx[img_name + element]]["features"][cat_size:cat_size+2].numpy()
        order = basic_ro(points)
        with open(out_folder + "/" + img_name + ".pickle", 'wb') as fh:
            pickle.dump((order,z), fh, protocol=pickle.HIGHEST_PROTOCOL)
        Fst[img_name] = metrics.spearman_footrule_distance(
            list(range(len(z))), list(order)
        )
        Ktd[img_name] = {"val":0, "nomalized":0}
        Ktd[img_name]['val'],Ktd[img_name]['normalized'] = metrics.kendall_tau_distance( list(range(len(z))), list(order), both=True)
    N = len(Fst.values())
    f = sum(Fst.values())/N
    print("Avg F(s,t): {}".format(f))
    kv = 0
    kn = 0
    for i,v in Ktd.items():
        kv += v['val']
        kn += v['normalized']
        #print(v['val'])
    print("Avg K(s,t): {} normalzed: {}".format(kv/N,kn/N))
    


if __name__=='__main__':
    main()
