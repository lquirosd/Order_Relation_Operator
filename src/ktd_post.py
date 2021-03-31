from metrics import kendall_tau_distance as ktd
from metrics import spearman_footrule_distance as sfd
import numpy as np
import pickle
import glob
import sys
import decode

folder =sys.argv[1]
Ktd = {}
Sfd = {}
N = 0
A = np.zeros((2,2))
for dec in decode.DECODERS:
    d = dec(A)
    Ktd[d.id] = {}
    Sfd[d.id] = {}

for f in glob.glob(folder + '/*.pickle'):
    with open(f, 'rb') as fh:
        data = pickle.load(fh)
    #t = list(data[0])
    #s = list(range(0, len(data[0])))
    s = data[0]
    s = list(range(len(s)))
    for dec in decode.DECODERS:
        d = dec(data[1]['Prob'].copy())
        d.run()
        Ktd[d.id][f] = {"val":0, "normalized":0}
        Sfd[d.id][f] = 0
        #--- if len(s) == 1 there is no doubt about the RO
        if len(s) > 1:
            Ktd[d.id][f]['val'],Ktd[d.id][f]['normalized'] = ktd( s, list(d.best_path), both=True)
            Sfd[d.id][f] = sfd(s, list(d.best_path))
    N += 1
    #print(f)
    #print(ktd(s,t,both=True))

for dec in decode.DECODERS:
    d =dec(A)
    kv = 0
    kn = 0
    sd = 0
    for i,v in Ktd[d.id].items():
        kv += v['val']
        kn += v['normalized']
        sd += Sfd[d.id][i]
    #print(v['val'])
    print("{}: Avg K(s,t): {} normalzed: {} S(s,t): {}".format(d.id,kv/N,kn/N,sd/N), end=' ')
    #print("{}: Avg S(s,t): {}".format(d.id,sd/N))
print("")
