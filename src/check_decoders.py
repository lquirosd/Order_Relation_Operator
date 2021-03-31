import numpy as np
import pickle
import decode
import metrics
import glob
import sys
import time

Fst = {}
T = {}

fpath = sys.argv[1]
l = int(sys.argv[2])
if l < 0:
    l = None
P = np.ones(1)
for dec in decode.DECODERS:
    d = dec(P)
    Fst[d.id] = {}
    T[d.id] = {}

for f in glob.glob(fpath + "/*.pickle"):
    with open(f,'rb') as fh:
        data = pickle.load(fh)
    N = data[1]["Prob"].shape[0]
    #idx = np.random.choice(range(N), l if l<N else N, replace=False)
    #idx = sorted(idx)
    idx = list(range(l if l < N else N))
    P = data[1]["Prob"][idx][:,idx]
    #gt = list(range(len(data[0][0:l])))
    #gt = list(range(len(data[2][idx])))
    gt = list(range(l if l < N else N))
    if len(gt) != P.shape[0]:
        print(f)
        print(len(gt))
        print(gt)
        print(P.shape)
        print("ERROR")
    for dec in decode.DECODERS:
        i_time = time.time()
        d = dec(P.copy())
        d.run()
        if d.best_path is not None:
            Fst[d.id][f] = metrics.spearman_footrule_distance(gt, list(d.best_path))
        else:
            Fst[d.id][f] = 0
        T[d.id][f] = time.time() - i_time 


for k,v in Fst.items():
    #print("Test F(s,t) by {}: {} avg time: {}".format(k, sum(v.values())/len(v), sum(T[k].values())/len(T[k])))
    if l > 9 and k == 'hamiltonian':
        print("?0.0,?0.0",end=',')
    else:
        print("{},{}".format(sum(v.values())/len(v), sum(T[k].values())/len(T[k])), end=',')
print("")

