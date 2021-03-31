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
    level = sys.argv[4]
    assert(level in ['lines','regions','lines_hier'])
    #with open(gt_file, 'rb') as fh:
    #    data = pickle.load(fh)
    
    gt_data, gt_relatives, gt_order, _, _ = torch.load(gt_file)
    #print(data)
    
    id_to_idx = {}
    Sfd = {}
    Ktd = {}

    for idx, d in enumerate(gt_data):
        if level == 'lines_hier':
            id_to_idx[d["l0_parent"] + "_" + d["parent"] + d["id"]] = idx
        else:
            id_to_idx[d["parent"] + d["id"]] = idx


    for img_name in gt_order.keys():
        z = gt_order[img_name]
        points = np.zeros((len(z), 2))
        for j, element in enumerate(z):
            if level == 'regions':
                feats = gt_data[id_to_idx[img_name + element]]["features"]
                center = np.array([feats[-4]+(feats[-3]-feats[-4])/2,
                                   feats[-2]+(feats[-1]-feats[-2])/2]) 
                points[j] = center
            else:
                points[j] = gt_data[id_to_idx[img_name + element]]["features"][cat_size:cat_size+2].numpy()
        order = basic_ro(points)
        with open(out_folder + "/" + img_name + ".pickle", 'wb') as fh:
            pickle.dump((order,z), fh, protocol=pickle.HIGHEST_PROTOCOL)
        if len(z) > 1:
            Sfd[img_name] = metrics.spearman_footrule_distance(
                list(range(len(z))), list(order)
            )
            Ktd[img_name] = {"val":0, "normalized":0}
            Ktd[img_name]['val'],Ktd[img_name]['normalized'] = metrics.kendall_tau_distance( list(range(len(z))), list(order), both=True)
        else:
            Sfd[img_name] = 0
            Ktd[img_name] = {"val":0, "normalized":0}
    N = len(Sfd.values())
    f = sum(Sfd.values())/N
    #print("Avg F(s,t): {}".format(f))
    kv = 0
    kn = 0
    for i,v in Ktd.items():
        kv += v['val']
        kn += v['normalized']
        #print(v['val'])
    print("Avg K(s,t): {} normalzed: {} S(s,t): {}".format(kv/N,kn/N, f))
    


if __name__=='__main__':
    main()
