import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from W_Construct import KNN,norm_W
from data_loader import load_mat
import numpy as np
from sklearn import metrics
from layers import GraphConvolution
import warnings
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='SCSEL')
parser.add_argument('--epochs', '-te', type=int, default=20, help='number of train_epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='Adam learning rate')
parser.add_argument('--r', type=float, default=-1, help='Scalar to control the distribution of the weights')
parser.add_argument('--dataset', type=str, default='MSRCV1', help='choose a dataset')
args = parser.parse_args()
class GCN(nn.Module):
    def __init__(self, nfeat,nhid):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return x
class SCSEL(nn.Module):
    def __init__(self,n,n_cluster):
        super(SCSEL, self).__init__()
        self.l2 = nn.Linear(n,32*4)
        self.l3 = nn.Linear(16*4, n_cluster)
        self.l4 = nn.Linear(32*4,16*4)
        self.gcn = GCN(32*4, 32*4)
        self.weight = nn.Parameter(torch.full((n_view,), 1.0), requires_grad=True)
        self.num = n


    def forward(self,W):

        weight = F.softmax(self.weight)
        weight = torch.pow(weight, -1)
        W_new = torch.matmul(W, weight)
        hid = F.leaky_relu(self.l2(W_new))
        CF = self.gcn(hid, W_new)
        CF = F.leaky_relu(self.l4(hid + CF))
        CF = F.softmax(self.l3(CF),dim=1)

        return CF, W_new, hid


    def run(self,W,n_min,n_max,l1,l2):
            optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
            self.to(device)
            ACC=[]
            NMI = []
            PUR = []
            ARI = []
            OBJ = []
            for it in range(args.epochs):
                for i in range(100):
                    optimizer.zero_grad()
                    CF, W_new, hid = self(W)
                    loss_gr = torch.pow(torch.norm(W_new-CF@CF.t()),2)
                    loss_size1 = torch.pow(torch.norm(F.relu(n_min*torch.ones((n_cluster,1)).to(device)-torch.mm(CF.t(),torch.ones((self.num,1)).to(device)))),2)
                    loss_size2 = torch.pow(torch.norm(F.relu(torch.mm(CF.t(), torch.ones((self.num, 1)).to(device)) - n_max * torch.ones((n_cluster, 1)).to(device))), 2)
                    loss_sp = -torch.trace(torch.pow(CF.t() @ CF, 1 / 2))
                    loss = loss_gr+ l1*(loss_size1 +loss_size2)+loss_sp*l2
                    loss.backward(retain_graph=True)
                    optimizer.step()
                CF = CF.detach().cpu().numpy()

                y_pred = np.argmax(CF, axis=1) + 1
                Acc = acc(GT, y_pred)
                Nmi = metrics.normalized_mutual_info_score(GT, y_pred)
                Pur = purity_score(GT, y_pred)
                Ari = metrics.adjusted_rand_score(GT, y_pred)
                print(Acc,Nmi,Pur,Ari)
                ACC.append(Acc)
                NMI.append(Nmi)
                PUR.append(Pur)
                ARI.append(Ari)
                obj = np.power(np.linalg.norm(W_new.detach().cpu().numpy() - CF @ CF.T), 2)
                OBJ.append(obj)
                if Acc ==1:
                    break

            return CF,y_pred,ACC,NMI,PUR,ARI,OBJ
def triu(X):
    return torch.sum(torch.triu(X, diagonal=1))
def NE(y_pred):
    n = len(y_pred)
    ar, num = np.unique(y_pred, return_counts=True)
    c = len(ar)
    ne = 0
    for i in range(c):
        ne = ne + (num[i]/n)*np.log(num[i]/n)
    return (-1/np.log(c))*ne

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    r_ind,c_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(r_ind,c_ind)]) * 1.0 / y_pred.size

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner
    return metrics.accuracy_score(y_true, y_voted_labels)

def add_gaussian_noise(X, alpha=0.1, mode="std"):
    """
    X: numpy array OR torch tensor
    """

    # ===== 1. convert to tensor safely =====
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    else:
        X = X.float()

    # ===== 2. compute noise =====
    if mode == "std":
        noise_std = alpha * torch.std(X)
        noise = torch.randn_like(X) * noise_std
    elif mode == "fixed":
        noise = torch.randn_like(X) * alpha
    else:
        raise ValueError("Unknown mode")

    return X + noise

if __name__ =="__main__":
    X, GT = load_mat('MVCdata/{}.mat'.format(args.dataset))
    n_cluster = len(np.unique(GT))
    N = X[0].shape[0]
    GT = GT.reshape(np.max(GT.shape), )
    c = n_cluster
    n_view = len(X)
    m = 15
    B = torch.zeros((N, N, n_view))
    Max_list = [np.floor(1.2 * N / c), np.floor(1.4 * N / c), np.floor(1.6 * N / c), np.floor(1.8 * N / c)]
    Min_list = [np.floor(0.8 * N / c), np.floor(0.6 * N / c), np.floor(0.4 * N / c), np.floor(0.2 * N / c)]
    print(np.unique(GT, return_counts=True)[1])
    print(Max_list)
    print(Min_list)
    for i in range(n_view):
        A = norm_W(KNN(X[i],m))
        B[:, :, i] = torch.tensor(A, dtype=torch.float32)
    para = [0.1,1,10,100]

    for i in range(4):
        for j in range(4):
            A = []
            NM = []
            P = []
            AR = []
            for inter in range(10):
                t1 = time.time()
                model = SCSEL(N, n_cluster)
                CF,  y_pred,a,n,p,ar,OBJ = model.run(B.to(device),
                                                         torch.tensor(Min_list[i], dtype=torch.float32).to(device),
                                                         torch.tensor(Max_list[j], dtype=torch.float32).to(device),
                                                         para[3],para[3]
                                                         )
                t2 = time.time()
                print(t2 - t1)
                y_pred = np.argmax(CF, axis=1) + 1
                print(np.unique(y_pred, return_counts=True)[1],len(np.unique(y_pred, return_counts=True)[1]))
                A.append(np.max(a))
                NM.append(np.max(n))
                P.append(np.max(p))
                AR.append(np.max(ar))
            print(OBJ)
            print(A)
            print(NM)
            print(P)
            print(AR)
            print('ACC_mean: {}, ACC_std: {}'.format(np.array(A).mean(), np.array(A).std()))
            print('NMI_mean: {}, NMI_std: {}'.format(np.array(NM).mean(), np.array(NM).std()))
            print('Purity_mean: {}, Purity_std: {}'.format(np.array(P).mean(), np.array(P).std()))
            print('ARI_mean: {}, Purity_std: {}'.format(np.array(AR).mean(), np.array(AR).std()))


