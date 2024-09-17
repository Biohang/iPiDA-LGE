import pandas as pd
import numpy as np
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# evaluate results for 100 divided samples
result_path = '../compare_res/'
methods = ['iPiDA-LGE','iPiDA-PUL','iPiDA-GCN','CLPiDA','iPiDA-G','ETGPDA','PUTransGCN']
f_post = '_res.csv'

# evaluate results for leave-one-disease-out cv
#result_path = '../compare_res_disease/'
#methods = ['iPiDA-LGE','iPiDA-PUL','iPiDA-GCN','CLPiDA','iPiDA_G','ETGPDA','PUTransGCN']
#f_post = '_res_disease.csv'


auc_results = {}
aupr_results = {}
spe_results = {}
sen_results = {}
pre_results = {}
acc_results = {}
f1_results = {}


for method in methods:
    res_file = result_path + method + f_post  
    data = pd.read_csv(res_file)
    # Compute the mean and variance for each metric (AUC, AUPR, F1-score)
    auc_results[method] = {'mean': np.mean(data['AUC']), 'var': np.var(data['AUC'])}
    aupr_results[method] = {'mean': np.mean(data['AUPR']), 'var': np.var(data['AUPR'])}
    spe_results[method] = {'mean': np.mean(data['SPE']), 'var': np.var(data['SPE'])}
    sen_results[method] = {'mean': np.mean(data['SEN']), 'var': np.var(data['SEN'])}
    pre_results[method] = {'mean': np.mean(data['PRE']), 'var': np.var(data['PRE'])}
    acc_results[method] = {'mean': np.mean(data['ACC']), 'var': np.var(data['ACC'])}
    f1_results[method] = {'mean': np.mean(data['F1']), 'var': np.var(data['F1'])}


proposed_res_file = result_path + methods[0] + f_post
proposed_auc = pd.read_csv(proposed_res_file)['AUC']
proposed_aupr = pd.read_csv(proposed_res_file)['AUPR']
proposed_spe= pd.read_csv(proposed_res_file)['SPE']
proposed_sen = pd.read_csv(proposed_res_file)['SEN']
proposed_pre = pd.read_csv(proposed_res_file)['PRE']
proposed_acc = pd.read_csv(proposed_res_file)['ACC']
proposed_f1 = pd.read_csv(proposed_res_file)['F1']

p_value_auc = {}
p_value_aupr = {}
p_value_spe = {}
p_value_sen = {}
p_value_pre = {}
p_value_acc = {}
p_value_f1 = {}

for method in methods[1:]:
    res_file = result_path + method + f_post
    data = pd.read_csv(res_file)
    
    # Perform t-test (or any other statistical test)

    p_value_auc[method] = stats.ranksums(proposed_auc, data['AUC']).pvalue
    p_value_aupr[method] = stats.ranksums(proposed_aupr, data['AUPR']).pvalue
    p_value_spe[method] = stats.ranksums(proposed_spe, data['SPE']).pvalue
    p_value_sen[method] = stats.ranksums(proposed_sen, data['SEN']).pvalue
    p_value_pre[method] = stats.ranksums(proposed_pre, data['PRE']).pvalue
    p_value_acc[method] = stats.ranksums(proposed_acc, data['ACC']).pvalue
    p_value_f1[method] = stats.ranksums(proposed_f1, data['F1']).pvalue


print("\nAUC Results (Mean and Variance):", auc_results)
print("\nAUPR Results (Mean and Variance):", aupr_results)
print("\nSPE Results (Mean and Variance):", spe_results)
print("\nSEN Results (Mean and Variance):", sen_results)
print("\nPRE Results (Mean and Variance):", pre_results)
print("\nACC Results (Mean and Variance):", acc_results)
print("\nF1 Results (Mean and Variance):", f1_results)

print("\nP-values for AUC comparison:", p_value_auc)
print("\nP-values for AUPR comparison:", p_value_aupr)
print("\nP-values for SPE comparison:", p_value_spe)
print("\nP-values for SEN comparison:", p_value_sen)
print("\nP-values for PRE comparison:", p_value_pre)
print("\nP-values for ACC comparison:", p_value_acc)
print("\nP-values for F1 comparison:", p_value_f1)




#############################################################################################################
# Box plots for different methods on 100 independent test results and leave-one-disease-out cross validation
result_path = '../compare_res/'
methods = ['iPiDA-LGE','iPiDA-PUL','iPiDA-GCN','CLPiDA','iPiDA-G','ETGPDA','PUTransGCN']
f_post = '_res.csv'
file_0 = result_path + methods[0] + f_post
file_1 = result_path + methods[1] + f_post
file_2 = result_path + methods[2] + f_post
file_3 = result_path + methods[3] + f_post
file_4 = result_path + methods[4] + f_post
file_5 = result_path + methods[5] + f_post
file_6 = result_path + methods[6] + f_post
files = [file_0,file_1,file_2,file_3,file_4,file_5,file_6]
metrics = ['AUC','AUPR','F1','ACC']
data_frames = []
for file, method in zip(files, methods):
    df = pd.read_csv(file)
    df['Method'] = method
    data_frames.append(df)

df_all = pd.concat(data_frames)

for method in methods[1:]:
    plot_method_vs_method1(method)

def plot_method_vs_method1(method):
  df_filtered = df_all[df_all['Method'].isin(['iPiDA-LGE',method])]
  p_values = {}
  method1_data = df_filtered[df_filtered['Method'] == 'iPiDA-LGE']
  method_data = df_filtered[df_filtered['Method'] == method]
  #palette_colors = ['salmon','lightblue', 'lightgreen', 'gold']
  palette_colors = ['#FFDAB9', '#ADD8E6']
  plt.figure(figsize=(10, 6))
  sns.boxplot(x='Metric', y='Value', hue='Method', 
      data=df_filtered.melt(id_vars=['Method'], value_vars=metrics, var_name='Metric', value_name='Value'),
      palette= palette_colors)
  plt.xlabel('Metric')
  plt.ylabel('Value')
  plt.ylim(0.53,0.98)
  #plt.ylim(0.1,1.1)
  #plt.legend(loc='lower right')
  plt.legend(loc='upper right')
  plt.show()

plot_method_vs_method1(methods[1])
plot_method_vs_method1(methods[2])
plot_method_vs_method1(methods[3])
plot_method_vs_method1(methods[4])
plot_method_vs_method1(methods[5])
plot_method_vs_method1(methods[6])


