import os
import glob
import sys
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms
import pickle
import numpy as np

from text_line_dataset import PairsInMemoryDataset
import metrics
import decode
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

def ensamble(A):
    A  += np.finfo(np.float).eps
    A = (A + (1-A).T)/2
    for i in range(A.shape[0]):
        A[i,i] = np.finfo(np.float).eps
    T = (A>0.5).sum(axis=1)
    return T.argsort()[::-1]

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
    FstE = {}
    A = np.zeros(1)
    for d in decode.DECODERS:
        dec = d(A)
        Fst[dec.id] = {}
    #print(len(list(T.keys())))
    #for page, lines in votes.items():
    for page in T.keys():
        s = dataloader.dataset.order[page]
        if len(s) == 1:
            for dec in decode.DECODERS:
                d = dec(A)
                Fst[d.id][page] = 0
            t = [0]
            FstE[page] = 0
        else:
            for dec in decode.DECODERS:
                d = dec(T[page]['Prob'].copy())
                d.run()
                if d.best_path is not None:
                    Fst[d.id][page] = metrics.spearman_footrule_distance(
                            list(range(len(s))), list(d.best_path)
                            )
                else:
                    Fst[d.id][page] = 1
                
            t = ensamble(T[page]['Prob'].copy())
            FstE[page] = metrics.spearman_footrule_distance(
                list(range(len(s))), list(t)
            )
        # --- save results
        if out_folder:
            with open(os.path.join(out_folder, page + ".pickle"), "wb") as fh:
                pickle.dump((t, T[page], s), fh)
    for k,v in Fst.items():
        print("Sp. FD: " + k + ": ", sum(v.values()) / len(v))

    return (sum(FstE.values()) / len(FstE), FstE)


def main():
    parser = argparse.ArgumentParser(description='MLP classifier for Pair-wise Reading order')
    parser.add_argument('--seed', 
            type=float, 
            default=datetime.datetime.now().microsecond,
            help='Random Seed',)
    parser.add_argument('--learning_rate',
            type=float,
            default=0.001,
            help='Learning Rate',)
    parser.add_argument('--batch_size',
            type=int,
            default=15000,
            help='Number samples per batch',)
    parser.add_argument('--epochs',
            type=int,
            default=3000,
            help='Number of training epochs',)
    parser.add_argument('--max_nondecreasing_epochs',
            type=int,
            default=300,
            help='Max number of non-decreasing epochs',)
    parser.add_argument('--evaluate_rate',
            type=int,
            default=300,
            help='Evaluate Validation set each number of epochs',)
    parser.add_argument('--echo_rate',
            type=int,
            default=1,
            help='Rate of info displayed',)
    parser.add_argument('--out_folder',
            type=str,
            default='./',
            help='Output Folder',)
    parser.add_argument('--tr_data',
            type=str,
            default='./',
            help='Pointer to training data files',)
    parser.add_argument('--val_data',
            type=str,
            default='./',
            help='Pointer to validation data files',)
    parser.add_argument('--te_data',
            type=str,
            default='./',
            help='Pointer to test data files',)
    parser.add_argument('--categories',
            type=str,
            nargs='+',
            help='List of categories (classes) to be used',)
    parser.add_argument('--exp_id',
            type=str,
            default='',
            help='Id assigned to the experiment.',)
    parser.add_argument('--force_regenerate',
            action='store_true',
            help='Force dataloader to re-generate samples.')
    parser.add_argument('--da',
            action='store_true',
            help='Use data augmentation',)
    parser.add_argument('--do_soft_val',
            type=bool,
            default=False,
            help='do soft Validation istead of full pairs',)
    parser.add_argument('--level',
            type=str,
            default='line',
            choices=['line', 'region'],
            help='do soft Validation istead of full pairs',)
    parser.add_argument('--hierarchical',
            action='store_true',
            help='Use hierarchical training at line level',)



    args = parser.parse_args()
    #seed = datetime.datetime.now().microsecond
    #seed = 5
    print(args)
    torch.manual_seed(args.seed)
    # --- Main hyperparameters
    #learning_rate = 0.001
    #epochs = 3
    #max_nondecreasing_epochs = 300
    #batch_size = 15000
    #echo_rate = 1
    #evaluate_rate = 4000
    #evaluate_rate = epochs + 1 if do_soft_val else evaluate_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #out_folder = sys.argv[1]
    #tr_data = sys.argv[2]
    tr_processed_data = os.path.join(args.tr_data, "processed")
    #val_data = sys.argv[3]
    val_processed_data = os.path.join(args.val_data, "processed")
    #te_data = sys.argv[4]
    te_processed_data = os.path.join(args.te_data, "processed")
    #categories = sys.argv[5].split()
    #exp_id = sys.argv[6]
    args.out_folder = os.path.join(args.out_folder, args.exp_id)
    #da = bool(int(sys.argv[7]))
    #try:
    #    do_soft_val = bool(int(sys.argv[8]))
    #except:
    #    do_soft_val = False
    # --- Get Data
    #categories = ["$pag", "$tip", "$par", "$not", "$nop", "$pac"]
    #categories = ["TextRegion", "TableRegion"]
    #categories = ['paragraph', 'paragraph_2', 'marginalia', 'page-number', 'table', 'table_2']
    #categories = ['paragraph', 'paragraph_2', 'marginalia', 'page-number']
    if args.da:
        mask = torch.zeros(2 * (len(args.categories) + 6), dtype=torch.bool)
        mask[0 : len(args.categories)] = True
        mask[6 + len(args.categories) : 6 + 2 * len(args.categories)] = True
        tr_transform = tv_transforms.Compose([RandomShift(mask=mask)])
    else:
        tr_transform = None
    tr_dataset = PairsInMemoryDataset(
        args.tr_data,
        set_id="train",
        processed_data=tr_processed_data,
        categories=args.categories,
        transform=tr_transform,
        force_regenerate=args.force_regenerate,
        level = args.level,
        hierarchical = args.hierarchical if args.level == 'line' else False
    )

    tr_dataloader = DataLoader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    )

    num_features = tr_dataset.get_num_features()
    hidden_size = num_features * 2
    # --- val data
    val_transform = None
    val_dataset = PairsInMemoryDataset(
        args.val_data,
        set_id="val",
        processed_data=val_processed_data,
        categories=args.categories,
        transform=val_transform,
        force_regenerate=args.force_regenerate,
        soft_val=args.do_soft_val,
        level = args.level,
        hierarchical = args.hierarchical if args.level == 'line' else False
    )
    print("Train num. samples: ", len(tr_dataset))
    print("Val num. pairs: ", len(val_dataset))

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    )

    # --- set model
    model = MLP(num_features, hidden_size, 1).to(device)
    # --- set criterion
    criterion = torch.nn.BCELoss().to(device)
    # --- set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # --- train the model
    best_val_loss = 10000
    best_epoch = 0

    mkdir(args.out_folder)

    for epoch in range(args.epochs):
        t_loss = train(model, tr_dataloader, criterion, optimizer, device)
        v_loss = validate(model, val_dataloader, criterion, device)
        v_loss = t_loss
        if not epoch % args.echo_rate:
            print(
                "Epoch {} : train-loss: {} val-loss: {}".format(
                    epoch, t_loss, v_loss
                )
            )
        if best_val_loss > v_loss:
            best_val_loss = v_loss
            best_epoch = epoch
            torch.save(
                model.state_dict(), 
                os.path.join(args.out_folder, "model.pth")
            )

        if (epoch - best_epoch) == args.max_nondecreasing_epochs:
            print(
                "Loss DID NOT decrease after {} consecutive epochs.".format(
                    args.max_nondecreasing_epochs
                )
            )
            break
        if ((epoch + 1) % args.evaluate_rate) == 0:
            val_eval = eval_sort(model, val_dataloader, device)
            print("Val F(s,t) after {} epochs: {}".format(epoch, val_eval[0]))

    # --- eval the model
    te_dataset = PairsInMemoryDataset(
        args.te_data,
        set_id="test",
        processed_data=te_processed_data,
        categories=args.categories,
        transform=None,
        force_regenerate=args.force_regenerate,
        level = args.level,
        hierarchical = args.hierarchical if args.level == 'line' else False
    )

    te_dataloader = DataLoader(
        te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
    )
    results_dir = os.path.join(args.out_folder ,"test")
    if best_epoch != epoch:
        model.load_state_dict(torch.load(
            os.path.join(args.out_folder, "model.pth")
            ))

    mkdir(results_dir)

    te_loss = eval_sort(model, te_dataloader, device, out_folder=results_dir)
    print("Test F(s,t) after Training: {}".format(te_loss[0]))


if __name__ == "__main__":
    main()
