from metrics import kendall_tau_distance as ktd
import numpy
import pickle
import glob
import sys

folder =sys.argv[1]
Ktd = {}
N = 0
for f in glob.glob(folder + '/*.pickle'):
    with open(f, 'rb') as fh:
        data = pickle.load(fh)
    t = list(data[0])
    s = list(range(0, len(data[0])))
    Ktd[f] = {"val":0, "nomalized":0}
    Ktd[f]['val'],Ktd[f]['normalized'] = ktd( s, t, both=True)
    N += 1
    #print(f)
    #print(ktd(s,t,both=True))
kv = 0
kn = 0
for i,v in Ktd.items():
    kv += v['val']
    kn += v['normalized']
    #print(v['val'])
print("Avg K(s,t): {} normalzed: {}".format(kv/N,kn/N))
