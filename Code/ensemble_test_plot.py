import numpy as np  
import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import evaluation_fun


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

#########################################################################################
# ensemble and test for iPiDA-L and iPiDA-G for the example independent test set
file_name = './data'
adjPD = np.load(file_name + '/adjPD.npy')
PD_ben_ind_label = np.load(file_name + '/PD_ben_ind_label.npy')
test_index = np.where((PD_ben_ind_label == -1) | (PD_ben_ind_label == -10))
test_label=np.load(file_name + "/test_label.npy")
#test_label = adjPD[test_index[0],test_index[1]]

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

SPE,SEN,PRE,ACC,F1_score = evaluation_fun.compute_metrics(score_w1,test_label)
SPE,SEN,PRE,ACC,F1_score = evaluation_fun.compute_metrics(results_global,test_label)

################################################################################################
#plot 2D scatter figure for the prediction results
x=test_label
y_local=results_local_norm
y_global=results_global_norm
y_both=score_w1
df_local = pd.DataFrame({"label":x,"predicted_results":y_local})
df_global = pd.DataFrame({"label":x,"predicted_results":y_global})
df_both = pd.DataFrame({"label":x,"predicted_results":y_both})

pal2 = sns.color_palette("CMRmap")
with sns.axes_style("white"):
    sns.jointplot(x='label', y='predicted_results',data=df_local,kind='kde',space=0,shade=True,joint_kws=dict(alpha =0.9),marginal_kws=dict(shade=True),cmap='Blues')
    sns.jointplot(x='label', y='predicted_results',data=df_global,kind='kde',space=0,shade=True,joint_kws=dict(alpha =0.9),marginal_kws=dict(shade=True),cmap='Blues')
    sns.jointplot(x='label', y='predicted_results',data=df_both,kind='kde',space=0,shade=True,joint_kws=dict(alpha =0.9),marginal_kws=dict(shade=True),cmap='Blues')

plt.show()

#AUC,AUPR Curve
precision_both, recall_both, thresholds = precision_recall_curve(test_label,score_w1)
fpr_both, tpr_both, _ = roc_curve(test_label,score_w1)
auc_both = roc_auc_score(test_label,score_w1)
aupr_both = average_precision_score(test_label,score_w1)
precision_local, recall_local, thresholds = precision_recall_curve(test_label,results_local_norm)
fpr_local, tpr_local, _ = roc_curve(test_label,results_local_norm)
auc_local= roc_auc_score(test_label,results_local_norm)
aupr_local = average_precision_score(test_label,results_local_norm)
precision_global, recall_global, thresholds = precision_recall_curve(test_label,results_global_norm)
fpr_global, tpr_global, _ = roc_curve(test_label,results_global_norm)
auc_global= roc_auc_score(test_label,results_global_norm)
aupr_global = average_precision_score(test_label,results_global_norm)

#plot auc
interval = 40
fpr_both_inter = fpr_both[::interval]
tpr_both_inter = tpr_both[::interval]
fpr_local_inter = fpr_local[::interval]
tpr_local_inter = tpr_local[::interval]
fpr_global_inter = fpr_global[::interval]
tpr_global_inter = tpr_global[::interval]
plt.plot([0, 1], [0, 1], linestyle='--', color='lightgray')
plt.plot(fpr_local_inter, tpr_local_inter, label=f'iPiDA-L (AUC = {auc_local:.4f})',marker='+',color=(129/255,179/255,169/255))
plt.plot(fpr_global_inter, tpr_global_inter, label=f'iPiDA-G (AUC = {auc_global:.4f})',marker='x',color=(243/255,200/255,70/255))
plt.plot(fpr_both_inter, tpr_both_inter, label=f'iPiDA-LGE (AUC = {auc_both:.4f})',marker='o',color=(35/255,100/255,168/255),markerfacecolor='none')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

#plot aupr
interval = 95
recall_local_inter = recall_local[::interval]
precision_local_inter = precision_local[::interval]
precision_global_inter = precision_global[::interval]
recall_global_inter = recall_global[::interval]
recall_both_inter = recall_both[::interval]
precision_both_inter = precision_both[::interval]
plt.plot([1, 0], [0.5, 1], linestyle='--', color='lightgray')
plt.plot(recall_local, precision_local, label=f'iPiDA-L (AUPR = {aupr_local:.4f})',color=(129/255,179/255,169/255))
plt.plot(recall_global, precision_global, label=f'iPiDA-G (AUPR = {aupr_global:.4f})',color=(243/255,200/255,70/255))
plt.plot(recall_both, precision_both, label=f'iPiDA-LGE (AUPR = {aupr_both:.4f})',color=(35/255,100/255,168/255))
plt.plot(recall_local_inter, precision_local_inter,linestyle= 'None',marker='+',markeredgecolor=(129/255,179/255,169/255))
plt.plot(recall_global_inter, precision_global_inter,linestyle= 'None',marker='x',markeredgecolor=(243/255,200/255,70/255))
plt.plot(recall_both_inter, precision_both_inter,linestyle= 'None',marker='o',markerfacecolor='none', markeredgecolor=(35/255,100/255,168/255))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.show()

#################################################################################################
# test for leave_one_disease_out cv  
evaluation_fun.evaluate_new_CV('../compare_res_disease','../data/random_disease/PD_ben_ind_disease','iPiDA-L','disease',19)
evaluation_fun.evaluate_new_CV('../compare_res_disease','../data/random_disease/PD_ben_ind_disease','iPiDA_G','disease',19)
evaluation_fun.evaluate_new_CV('../compare_res_disease','../data/random_disease/PD_ben_ind_disease','iPiDA-LGE','disease',19)
evaluation_fun.evaluate_new_CV('../compare_res_disease','../data/random_disease/PD_ben_ind_disease','iPiDA-PUL','disease',19)
evaluation_fun.evaluate_new_CV('../compare_res_disease','../data/random_disease/PD_ben_ind_disease','iPiDA-GCN','disease',19)
evaluation_fun.evaluate_new_CV('../compare_res_disease','../data/random_disease/PD_ben_ind_disease','ETGPDA','disease',19)
evaluation_fun.evaluate_new_CV('../compare_res_disease','../data/random_disease/PD_ben_ind_disease','CLPiDA','disease',19)
evaluation_fun.evaluate_new_CV('../compare_res_disease','../data/random_disease/PD_ben_ind_disease','PUTransGCN','disease',19)
###################################################
# test for 100 constructed independent test sets
evaluation_fun.evaluate_random_divide('../compare_res','../data/random_PD/PD_ben_ind_label','iPiDA-LGE',100)
evaluation_fun.evaluate_random_divide('../compare_res','../data/random_PD/PD_ben_ind_label','iPiDA-L',100)
evaluation_fun.evaluate_random_divide('../compare_res','../data/random_PD/PD_ben_ind_label','iPiDA-PUL',100)
evaluation_fun.evaluate_random_divide('../compare_res','../data/random_PD/PD_ben_ind_label','iPiDA-G',100)
evaluation_fun.evaluate_random_divide('../compare_res','../data/random_PD/PD_ben_ind_label','iPiDA-GCN',100)
evaluation_fun.evaluate_random_divide('../compare_res','../data/random_PD/PD_ben_ind_label','ETGPDA',100)
evaluation_fun.evaluate_random_divide('../compare_res','../data/random_PD/PD_ben_ind_label','PUTransGCN',100)
evaluation_fun.evaluate_random_divide('../compare_res','../data/random_PD/PD_ben_ind_label','CLPiDA',100)