# iPiDA-LGE: A Local and Global Graph Ensemble Learning Framework for Identifying PiRNA-Disease Associations

This repository contains the source code used in our paper titled iPiDA-LGE: A Local and Global Graph Ensemble Learning Framework for Identifying PiRNA-Disease Associations. The code is implemented to realize the proposed predictor, and the dataset and tutorials are also provided to assist users in utilizing the code.

# Introduction
In this study, we propose a novel computational method named iPiDA-LGE for identifying piRNA-disease associations. iPiDA-LGE comprises two graph convolutional neural network modules based on local and global piRNA-disease graphs, aimed at capturing specific and general features of piRNA-disease pairs. Additionally, it integrates their refined and macroscopic inferences to derive the final prediction result. The experimental results show that iPiDA-LGE effectively leverages the advantages of both local and global graph learning, thereby achieving stronger pair representation and superior prediction performance.
![image](https://github.com/Biohang/iPiDA-LGE/blob/main/Image/Fig1.jpg)  
**Figure.1**. The framework of iPiDA-LGE.

# Datasets 
----- piRNA_information.xlsx：list of piRNA names and piRNAs' corresponding number in the dataset   
----- disease_information.xlsx: list of disease names and diseases' corresponding number  
----- piSim.npy: piRNA-piRNA similarity matrix  
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
#####Training for local piRNA-disease graph
'python -u main.py \ 
            --hop={} \ ;                 # hop number of extracting local graph (default=2)  
            --lr ={} \ ;                 # learning rate (default=1e-3)  
            --epochs ={} \ ;          # number of epochs to train  (default=20)  
            '.format(hop, lr, epochs) 

######Integrating and evaluating of local and global prediction result
python ensemble.py 
