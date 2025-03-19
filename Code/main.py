import torch
import numpy as np
import sys, copy, math, time, pdb, warnings, traceback
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import pylab
import random
import argparse
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_functions import *
from train_eval import *
from models import *
import evaluation_fun
import traceback
import warnings
import sys
import matplotlib.pyplot as plt
import networkx as nx
import json

# assign cuda device ID
device = torch.device('cuda:1')



# used to traceback which code cause warnings, can delete
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


def logger(info, model, optimizer):
    epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
    with open(os.path.join('./model_save/', 'log.txt'), 'a') as f:
        f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
            epoch, train_loss, test_rmse))
    if type(epoch) == int and epoch % args.save_interval == 0:
        print('Saving model states...')
        model_name = os.path.join('./model_save/', 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(
            './model_save/', 'optimizer_checkpoint{}.pth'.format(epoch)
        )
        if model is not None:
            torch.save(model.state_dict(), model_name)



# Arguments
parser = argparse.ArgumentParser(description='iPiDA-L')
# general settings
parser.add_argument('--no-train', action='store_true', default=False,
                    help='if set, skip the training')
parser.add_argument('--debug', action='store_true', default=False,
                    help='turn on debugging mode which uses a small number of data')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                    help='seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                    valid only for ml_1m and ml_10m')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--save-interval', type=int, default=20,
                    help='save model states every # epochs ')
parser.add_argument('--data-name', default='ml_100k', help='dataset name')
parser.add_argument('--data-appendix', default='',
                    help='what to append to save-names when saving datasets')
parser.add_argument('--testing', action='store_true', default=False,
                    help='if set, use testing mode which splits all ratings into train/test;\
                    otherwise, use validation model which splits all ratings into \
                    train/val/test and evaluate on val only')
parser.add_argument('--save-appendix', default='',
                    help='what to append to save-names when saving results')
parser.add_argument('--max-train-num', type=int, default=None,
                    help='set maximum number of train data to use')
parser.add_argument('--max-val-num', type=int, default=None,
                    help='set maximum number of val data to use')
parser.add_argument('--max-test-num', type=int, default=None,
                    help='set maximum number of test data to use')
# subgraph extraction settings
parser.add_argument('--hop', default=2, metavar='S',
                    help='hop number of extracting local graph')
parser.add_argument('--sample-ratio', type=float, default=1.0,
                    help='if < 1, subsample nodes per hop according to the ratio')
parser.add_argument('--max-nodes-per-hop', default=100,
                    help='if > 0, upper bound the # nodes per hop by another subsampling')
parser.add_argument('--use-features', action='store_true', default=True,
                    help='whether to use node features (side information)')
# edge dropout settings
parser.add_argument('--adj-dropout', type=float, default=0,
                    help='if not 0, random drops edges from adjacency matrix with this prob')
parser.add_argument('--force-undirected', action='store_true', default=False,
                    help='in edge dropout, force (x, y) and (y, x) to be dropped together')
# optimization settings
parser.add_argument('--continue-from', type=int, default=None,
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--lr-decay-step-size', type=int, default=50,
                    help='decay lr by factor A every B steps')
parser.add_argument('--lr-decay-factor', type=float, default=0.2,
                    help='decay lr by factor A every B steps')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=300, metavar='N',
                    help='batch size during training')
parser.add_argument('--test-freq',
                    type=int, default=20, metavar='N',
                    help='test every n epochs')
parser.add_argument('--ARR', type=float, default=0.00,
                    help='The adjacenct rating regularizer. If not 0, regularize the \
                    differences between graph convolution parameters W associated with\
                    adjacent ratings')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

random.seed(args.seed)
np.random.seed(args.seed)
args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)

rating_map = None



args.file_dir = os.path.dirname(os.path.realpath('__file__'))
parser.add_argument('--testing', action='store_true', default=False,
                    help='if set, use testing mode which splits all ratings into train/test;\
                    otherwise, use validation model which splits all ratings into \
                    train/val/test and evaluate on val only')

def import_data():
    file_path = os.path.dirname(__file__)
    file_name = os.path.abspath(os.path.join(file_path, '../data/'))
    PS_seq = np.load(file_name + '/piSim.npy')
    DS_doid = np.load(file_name + '/DiseaSim.npy')
    adjPD = np.load(file_name + '/adjPD.npy')
    PD_ben_ind_label = np.load(file_name + '/PD_ben_ind_label.npy')

    return PS_seq, DS_doid, adjPD, PD_ben_ind_label


PS_seq, DS_doid, adjPD, PD_ben_ind_label = import_data()


#independent
train_index = np.where((PD_ben_ind_label == 1) | (PD_ben_ind_label == -20))
test_index = np.where((PD_ben_ind_label == -1) | (PD_ben_ind_label == -10))
adj_train = np.copy(adjPD)
adj_train[test_index] = 0

train_A_csr = ssp.csr_matrix(adj_train)
train_label = np.array(adj_train[train_index])
train_pi_indices = range(0, 10149)
train_dis_indices = range(0, 19)

adj_test = np.zeros(adjPD.shape)
adj_test[test_index] = adjPD[test_index]
test_A_csr = ssp.csr_matrix(adj_test)
test_label = np.array(adj_test[test_index])
test_label_fold=np.array(adjPD[test_index])

#side features combination
pi_features = np.array(PS_seq)
dis_features = np.array(DS_doid)


class_values = [0,1]
data_combo = (args.data_name, args.data_appendix, val_test_appendix)
train_graphs = eval('MyDynamicDataset')(
    'data/{}{}/{}/train'.format(*data_combo),
    train_A_csr,
    train_index,
    train_label,
    args.hop,
    args.sample_ratio,
    args.max_nodes_per_hop,
    pi_features,
    dis_features,
    class_values,
    max_num=args.max_train_num
)


test_graphs = eval('MyDynamicDataset')(
   'data/{}{}/{}/test'.format(*data_combo),
    train_A_csr,
    test_index,
    test_label,
    args.hop,
    args.sample_ratio,
    args.max_nodes_per_hop,
    pi_features,
    dis_features,
    class_values,
    max_num=args.max_train_num
)

graph_loader = DataLoader(test_graphs, shuffle=False)
serializable_data = [data.__dict__ for data in graph_loader]


# train model
num_relations = len(class_values)
multiply_by = 1
n_features = pi_features.shape[1] + dis_features.shape[1]

model = LocalGL(
    train_graphs,
    latent_dim=[32,1],
    num_relations=num_relations,
    num_bases=0,
    regression=True,
    adj_dropout=args.adj_dropout,
    force_undirected=args.force_undirected,
    side_features=args.use_features,
    n_side_features=n_features,
    multiply_by=multiply_by
)

if not args.no_train:
  train_multiple_epochs(
      train_graphs,
      test_graphs,
      model,
      args.epochs,
      args.batch_size,
      args.lr,
      lr_decay_factor=args.lr_decay_factor,
      lr_decay_step_size=args.lr_decay_step_size,
      weight_decay=0,
      ARR=args.ARR,
      test_freq= args.test_freq,
  )

