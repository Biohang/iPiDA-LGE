This repository contains the source code used in our paper titled iPiDA-LGE: A Local and Global Graph Ensemble Learning Framework for Identifying PiRNA-Disease Associations. The code is implemented to realize the proposed predictor, and the dataset and tutorials are also provided to assist users in utilizing the code.

# Datasets 
----- MNDR_piRNA.xlsx：list of piRNA names and piRNAs' corresponding number in the dataset   
----- MNDR_disease.xlsx: list of disease names and diseases' corresponding number  
----- DiseaSim.npy: disease-disease similarity matrix  
----- adjPD.npy: piRNA-disease association matrix, a(i,j) represents the association between i-th piRNA and j-th disease  
----- PD_ben_ind_label.npy: benchmark dataset and independent dataset division, where the label 1 and -20 represent positive and negative samples in benchmark dataset, label -1 and -10 represent positive and negative samples in independent dataset

# Usage
##### Basic environment setup:  
----- python 3.8  
----- cuda 11.3  
----- pytorch 1.12.0  

##### Training and Testing 
----- The source code and tutorials for global graph learning module in iPiDA-LGE can be available at http://bliulab.net/iPiDA-SWGCN/tutorial/  
----- The training codes for local graph learning module in iPiDA-LGE include the scripts main.py and train_eval.py  
----- models.py records the graph neural network for local context graph learning in iPiDA-LGE  
----- util_function.py records the basic functional functions for constructing local context graphs for target piRNA-disease pairs  
----- ensemble.py records the integration and evaluation of local and global prediction result  


##### Running example
----- Training for local piRNA-disease graph  
'python -u main.py \  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--hop={} \ ;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # hop number of extracting local graph (default=2)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--lr ={} \ ;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# learning rate (default=1e-3)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--epochs ={} \ ;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# number of epochs to train  (default=20)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'.format(hop, lr, epochs) 

----- Integrating and evaluating of local and global prediction result  
python ensemble.py 
