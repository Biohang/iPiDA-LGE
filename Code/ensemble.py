import numpy as np  
import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

#import data
file_name = './data'
adjPD = np.load(file_name + '/adjPD.npy')
PD_ben_ind_label = np.load(file_name + '/PD_ben_ind_label.npy')
test_index = np.where((PD_ben_ind_label == -1) | (PD_ben_ind_label == -10))
test_label=np.load(file_name + "/test_label.npy")

epoch = 20
results_local = np.load(file_name + "/test_results_local_test" + str(epoch) +".npy")
results_g = np.load(file_name + "/results_global_test.npy")
results_global= results_g[test_index]
results_local_norm=normalization(results_local)
results_global_norm=normalization(results_global)
test_pos=np.where(test_label==1)
test_neg=np.where(test_label==0)
score=np.copy(results_global_norm)
score_w1=0.6*results_local_norm+results_global_norm*0.4

precision_both, recall_both, thresholds = precision_recall_curve(test_label,score_w1)
fpr_both, tpr_both, _ = roc_curve(test_label,score_w1)
auc_both = roc_auc_score(test_label,score_w1)
aupr_both = average_precision_score(test_label,score_w1)
