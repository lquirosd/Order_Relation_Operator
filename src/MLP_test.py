import os
import glob
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms
import pickle
import numpy as np

from text_line_dataset import TextLineInMemoryDataset
import metrics
from utils import mkdir, files_exist
from models import MLP
from transforms import RandomShift
import datetime


def train(model, dataloader, criterion, optimizer, device):
    """
    Train the model for 1 epoch
    """
    model.train()
    g_loss = 0
    for batch, sample in enumerate(dataloader):
        optimizer.zero_grad()
        x = sample["x"].to(device)
        t = sample["t"].to(device)
        y = model(x)
        loss = criterion(y.squeeze(), t)
        g_loss += loss.data
        loss.backward()
        optimizer.step()
    return g_loss / (batch + 1)


def validate(model, dataloader, criterion, device):
    """
    evaluate the model
    """
    model.eval()
    g_loss = 0
    for batch, sample in enumerate(dataloader):
        x = sample["x"].to(device)
        t = sample["t"].to(device)
        y = model(x)
        loss = criterion(y.squeeze(), t)
        g_loss += loss.data
    return g_loss / (batch + 1)


def voting(A):
    A  += np.finfo(np.float).eps
    A = A/(A+A.T)
    for i in range(A.shape[0]):
        A[i,i] = np.finfo(np.float).eps
    T = (A>0.5).sum(axis=1)
    return T.argsort()[::-1]

def phi(A):
    A  += np.finfo(np.float).eps
    A = A/(A+A.T)
    for i in range(A.shape[0]):
        A[i,i] = np.finfo(np.float).eps
    T = A.sum(axis=1)
    return T.argsort()[::-1]

def phi_prob(A):
    A  += np.finfo(np.float).eps
    A = A/(A+A.T)
    for i in range(A.shape[0]):
        A[i,i] = 1
    A = np.log(A)
    T = A.sum(axis=1)
    return T.argsort()[::-1]

def greddy(A):
    T = np.zeros(A.shape)
    F = np.zeros(A.shape)
    B = np.zeros(A.shape)
    N = A.shape[0]
    A  += np.finfo(np.float).eps
    A = A/(A+A.T)
    for i in range(A.shape[0]):
        A[i,i] = np.finfo(np.float).eps
    #A = np.log(A)
    #--- prob s>s' \forall s,s' \in S 
    S = A.prod(axis=1)
    #--- prob s<s' \forall s,s' \in S
    E = (1-A).prod(axis=1)
    F[0,:] = S
    B[-1,:] = E
    for t in range(0,N):
        F[t,:] = (F[t-1,:] * A).sum(axis=1)
    for t in range(0,T.shape[0]):
        for s in range(0,T.shape[0]):
            #T[pos, element]
            T[t,s] = T[t-1,:]+A[s,:].sum()


def eval_sort(model, dataloader, device, out_folder=None):
    """
    evaluate the results as sorting alg
    """
    model.eval()
    #votes = {}
    T = {}
    for batch, sample in enumerate(dataloader):
        x = sample["x"].to(device)
        t = sample["t"].to(device)
        y = model(x).squeeze()
        parent = sample["parent"]
        for i, p in enumerate(parent):
            #if not p in votes:
            if not p in T:
                #votes[p] = {}
                ne = len(dataloader.dataset.order[p])
                T[p] = {'Prob':np.zeros((ne,ne)),
                        'order':dataloader.dataset.order[p]}
                #print(p)
            #if not sample["z"][0][i] in votes[p]:
            #    votes[p][sample["z"][0][i]] = 0
            #if not sample["z"][1][i] in votes[p]:
            #    votes[p][sample["z"][1][i]] = 0
            #
            #if y[i] > 0.5:
            #    votes[p][sample["z"][0][i]] += 1 
            #else:
            #    votes[p][sample["z"][1][i]] += 1
            idxi = T[p]['order'].index(sample["z"][0][i])
            idxj = T[p]['order'].index(sample["z"][1][i])
            T[p]['Prob'][idxi,idxj] = y[i]
    Fst = {}
    Fst2 = {}
    Fst3 = {}
    #print(len(list(T.keys())))
    #for page, lines in votes.items():
    for page in T.keys():
        s = dataloader.dataset.order[page]
        #t = [k for k, v in sorted(lines.items(), key=lambda item: item[1])]
        #t = t[::-1]
        t = voting(T[page]['Prob'])
        t2 = phi(T[page]['Prob'])
        t3 = phi_prob(T[page]['Prob'])
        #print(page)
        #print(T[p]['Prob'].shape)
        #print(len(T[p]['order']))
        #print(len(list(t)))
        #print(len(s))
        #print(len(t))
        #print(len(list(range(len(s)))))
        #print(s[0])
        #print(T[p]['order'][0])
        Fst[page] = metrics.spearman_footrule_distance(
            list(range(len(s))), list(t)
        )
        Fst2[page] = metrics.spearman_footrule_distance(
            list(range(len(s))), list(t2)
        )
        Fst3[page] = metrics.spearman_footrule_distance(
            list(range(len(s))), list(t3)
        )
        #Fst[page] = metrics.spearman_footrule_distance(
        #    list(range(len(s))), [t.index(i) for i in s]
        #)
        #print(Fst[page], " ---- ", a)
        # --- save results
        if out_folder:
            with open(os.path.join(out_folder, page + ".pickle"), "wb") as fh:
                pickle.dump((t, T[page]), fh)

    print("Test F(s,t) voting: {}".format(sum(Fst.values()) / len(Fst)))
    print("Test F(s,t) Sum: {}".format(sum(Fst2.values()) / len(Fst2)))
    print("Test F(s,t) Prod: {}".format(sum(Fst3.values()) / len(Fst3)))
    return (sum(Fst.values()) / len(Fst), Fst)


def main():
    seed = datetime.datetime.now().microsecond
    #seed = 5
    torch.manual_seed(seed)
    print("Seed: ",seed)
    # --- Main hyperparameters
    learning_rate = 0.001
    epochs = 3000
    max_nondecreasing_epochs = 300
    batch_size = 15000
    echo_rate = 1
    evaluate_rate = 4000
    #evaluate_rate = epochs + 1 if do_soft_val else evaluate_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_folder = sys.argv[1]
    tr_data = sys.argv[2]
    tr_processed_data = os.path.join(tr_data, "processed")
    val_data = sys.argv[3]
    val_processed_data = os.path.join(val_data, "processed")
    te_data = sys.argv[4]
    te_processed_data = os.path.join(te_data, "processed")
    categories = sys.argv[5].split()
    exp_id = sys.argv[6]
    out_folder = os.path.join(out_folder, exp_id)
    da = bool(int(sys.argv[7]))
    try:
        do_soft_val = bool(int(sys.argv[8]))
    except:
        do_soft_val = False
    # --- Get Data
    #categories = ["$pag", "$tip", "$par", "$not", "$nop", "$pac"]
    #categories = ["TextRegion", "TableRegion"]
    #categories = ['paragraph', 'paragraph_2', 'marginalia', 'page-number', 'table', 'table_2']
    #categories = ['paragraph', 'paragraph_2', 'marginalia', 'page-number']
    if da:
        mask = torch.zeros(2 * (len(categories) + 6), dtype=torch.bool)
        mask[0 : len(categories)] = True
        mask[6 + len(categories) : 6 + 2 * len(categories)] = True
        tr_transform = tv_transforms.Compose([RandomShift(mask=mask)])
    else:
        tr_transform = None
    tr_dataset = TextLineInMemoryDataset(
        tr_data,
        set_id="train",
        processed_data=tr_processed_data,
        categories=categories,
        transform=tr_transform,
        force_regenerate=False,
    )

    tr_dataloader = DataLoader(
        tr_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    )

    num_features = tr_dataset.get_num_features()
    hidden_size = num_features * 2
    # --- set model
    model = MLP(num_features, hidden_size, 1).to(device)
    # --- set criterion
    criterion = torch.nn.BCELoss().to(device)
    # --- set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # --- train the model
    best_val_loss = 10000
    best_epoch = 0

    mkdir(out_folder)

    # --- eval the model
    te_dataset = TextLineInMemoryDataset(
        te_data,
        set_id="test",
        processed_data=te_processed_data,
        categories=categories,
            transform=None,
        force_regenerate=False,
    )

    te_dataloader = DataLoader(
        te_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
    )
    results_dir = os.path.join(out_folder ,"test")
    epoch = 100
    if best_epoch != epoch:
        model.load_state_dict(torch.load(
            os.path.join(out_folder, "model.pth")
            ))

    mkdir(results_dir)

    te_loss = eval_sort(model, te_dataloader, device, out_folder=results_dir)
    print("Test F(s,t) after Training: {}".format(te_loss[0]))
    print(len(te_dataset))


if __name__ == "__main__":
    main()
