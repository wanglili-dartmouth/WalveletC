import networkx as nx
from WaveletC import WaveletC
import pandas as pd
from utils import load_graph, load_features, load_graphs, save_embedding
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

def test(labels,embedding_dict):
    embedding=np.array([(embedding_dict[i]) for i in range(len(labels))])
    all_score=[]
    for K in range(10):
        X_train, X_test, y_train, y_test=train_test_split(   embedding, labels, test_size=0.2, random_state=K)
        print(len(X_train))
        print(len(X_test))
        clf = LogisticRegression(solver="saga").fit(X_train, y_train)
        all_score.append(roc_auc_score(y_test, clf.decision_function(X_test)))
    print(np.mean(all_score), file = sample)
    print(np.std(all_score), file = sample)
    print(np.std(all_score)/np.sqrt(len(all_score)), file = sample)
   
            
sample = open("results"+'.out', 'w') 
for dataset in ["git","deezer","twitch","reddit"]: 
    graphs = load_graphs(dataset+"_edges.json")
    labels = pd.read_csv(dataset+'_target.csv', index_col=0)
    labels=labels.values
    
    model = WaveletC()
    model.fit(graphs)
    embedding = model.get_embedding()
    print(dataset, file = sample)
    print(embedding.shape, file = sample)
    test(labels,embedding)
    sample.flush()
    