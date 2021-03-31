from metrics import kendall_tau_distance as ktd
from metrics import spearman_footrule_distance as sfd
import numpy as np
import pickle
import glob
import sys
import os
import decode


reg_folder =sys.argv[1]
lin_folder =sys.argv[2]
has_prob = 0 if sys.argv[3] == '0' else 1
Ktd = {}
Sfd = {}
N = 0
if has_prob:
    A = np.zeros((2,2))
    for dec in decode.DECODERS:
        d = dec(A)
        Ktd[d.id] = {}
        Sfd[d.id] = {}




for f in glob.glob(reg_folder + '/*.pickle'):
    with open(f, 'rb') as fh:
        r_data = pickle.load(fh)
    #t = list(data[0])
    #s = list(range(0, len(data[0])))
    l = os.path.basename(f)
    #if l == "170025120000003,0272.pickle":
    #    pass
    #else:
    #    continue
    r_s = r_data[0]
    r_s = list(range(len(r_s)))
    if has_prob:
        for dec in decode.DECODERS:
            d = dec(r_data[1]['Prob'].copy())
            d.run()
            Ktd[d.id][f] = {"val":0, "normalized":0, "summed":0}
            Sfd[d.id][f] = 0
            #--- if len(s) == 1 there is no doubt about the RO
            if len(r_s) > 1:
                kr, krn = ktd(r_s, list(d.best_path), both=True)
            else:
                kr, krn = 0,0
            Ktd[d.id][f]["summed"] +=  kr
            #    Sfd[d.id][f] = sfd(r_s, list(d.best_path))
            #    r_z = [x for _,x in sorted(zip(d.best_path, r_data[2]))]
            #else:
            #    r_z = r_data[2]
            r_z = [x for _,x in sorted(zip(d.best_path, r_data[2]))]
            l_g = []
            p_l = 0
            l_z = [None for z in r_z]
            l_h = []
            #kk = False
            for z in r_z:
                try:
                    with open(lin_folder + "/" + z+"_"+l, 'rb') as fh:
                        l_data = pickle.load(fh)
                except:
                    #kk = True
                    l_z[r_data[2].index(z)] = []
                    continue
                l_s = list(range(len(l_data[0])))
                l_d = dec(l_data[1]['Prob'].copy())
                l_d.run()
                if len(l_s) > 1:
                    kl, kln = ktd(l_s, list(l_d.best_path), both=True)
                else:
                    kl, kln = 0,0
                Ktd[d.id][f]["summed"] += kl
                r_id = r_data[2].index(z)
                l_g.extend([x+p_l for x in l_d.best_path])
                p_l += len(l_s)
                l_z[r_data[2].index(z)] = l_data[2]
                l_h.extend([x for _,x in sorted(zip(l_d.best_path,l_data[2]))])
            try:
                l_z = [item for sublist in l_z for item in sublist]
            except:
                print(l_z)
                exit()
            if len(l_g) > 1:
                #g_s = list(range(len(l_g)))
                g_s = list(range(len(l_z)))
                l_g = [l_h.index(x) for x in l_z]
                #if kk:
                #    print(g_s, sorted(l_g))
                #if g_s == l_g:
                #    pass
                #else:
                #    print(g_s, l_g)
                Ktd[d.id][f]['val'],Ktd[d.id][f]['normalized'] = ktd(g_s, l_g, both=True)
                Sfd[d.id][f] = sfd(g_s, l_g)
            #print(kr,Ktd[d.id][f]["summed"])
        N += 1
    else:
        r_z = [x for _,x in sorted(zip(r_data[0], r_data[1]))]
        p_l = 0
        Ktd[f] = {"val":0, "normalized":0, "summed":0}
        Sfd[f] = 0
        l_g = []
        l_z = [None for z in r_z]
        l_h = []
        if len(r_s) > 1:
            kr, krn = ktd(r_s, list(r_data[0]), both=True)
        else:
            kr, krn = 0,0
        Ktd[f]["summed"] +=  kr
        for z in r_z:
            try:
                with open(lin_folder + "/" + z+"_"+l, 'rb') as fh:
                    l_data = pickle.load(fh)
            except:
                #kk = True
                l_z[r_data[1].index(z)] = []
                continue
            l_z[r_data[1].index(z)] = l_data[1]
            l_h.extend([x for _,x in sorted(zip(l_data[0],l_data[1]))])
            l_s = list(range(len(l_data[0])))
            l_g.extend([x+p_l for x in l_data[0]])
            p_l += len(l_s)
            if len(l_s) > 1:
                kl, kln = ktd(l_s, list(l_data[0]), both=True)
            else:
                kl, kln = 0,0
            Ktd[f]["summed"] += kl
            #print(p_l)
        l_z = [item for sublist in l_z for item in sublist]
        if len(l_g)>1:
            #g_s = list(range(len(l_g)))
            g_s = list(range(len(l_z)))
            l_g = [l_h.index(x) for x in l_z]
            #if kk:
            #    print(g_s, sorted(l_g))
            #print(g_s, l_g)
            Ktd[f]['val'],Ktd[f]['normalized'] = ktd(g_s, l_g, both=True)
            Sfd[f] = sfd(g_s, l_g)
        N += 1

if has_prob:
    for dec in decode.DECODERS:
        d =dec(A)
        kv = 0
        kn = 0
        ks = 0
        sd = 0
        for i,v in Ktd[d.id].items():
            kv += v['val']
            ks += v['summed']
            kn += v['normalized']
            sd += Sfd[d.id][i]
        #print(v['val'])
        print("{}: Avg K(s,t): {} normalzed: {} summed: {} S(s,t): {}".format(d.id,kv/N,kn/N,ks/N,sd/N), end=' ')
        #print("{}: Avg S(s,t): {}".format(d.id,sd/N))
else:
    kv = 0
    kn = 0
    ks = 0
    sd = 0
    for i,v in Ktd.items():
        kv += v['val']
        ks += v['summed']
        kn += v['normalized']
        sd += Sfd[i]
    print("{}: Avg K(s,t): {} normalzed: {} summed: {} S(s,t): {}".format("TBLR",kv/N,kn/N,ks/N,sd/N), end=' ')

print("")
