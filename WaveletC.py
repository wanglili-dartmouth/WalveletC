import math
from typing import List
import numpy as np
import networkx as nx
import scipy.sparse as sparse
from karateclub.estimator import Estimator
from graphwave import graphwave_alg
from tqdm import tqdm
from sklearn.preprocessing import normalize
import time
class WaveletC(Estimator):


    def __init__(self, order: int=5, eval_points: int=25,
                 theta_max: float=2.5, tau:float=0.5,seed: int=42, pooling: str="mean"):
        self.order = order
        self.eval_points = eval_points
        self.theta_max = theta_max
        self.seed = seed
        self.pooling = pooling
        self.tau=tau


    def _create_D_inverse(self, graph):

        index = np.arange(graph.number_of_nodes())
        values = np.array([1.0/graph.degree[node] for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse


    def _get_normalized_adjacency(self, graph):

        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat


    def _create_node_feature_matrix(self, graph):
        log_degree = np.array([math.log(graph.degree(node)+1) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        clustering_coefficient = np.array([nx.clustering(graph, node) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        X = np.concatenate([log_degree, clustering_coefficient], axis=1)
        return X


    def _calculate_WaveletC(self, graph):

        A_tilde = self._get_normalized_adjacency(graph)
        X = self._create_node_feature_matrix(graph)
        theta = np.linspace(0.01, self.theta_max, self.eval_points)
        X = np.outer(X, theta)
        X = X.reshape(graph.number_of_nodes(), -1)
        X = np.concatenate([np.cos(X), np.sin(X)], axis=1)
        X1=np.copy(X)
        X2=np.copy(X)
        feature_blocks = []
        A_tilde=A_tilde.toarray()
        tmp=np.copy(A_tilde)
        
        heat_print, taus = graphwave_alg(graph, np.linspace(0,100,25), taus=[self.tau], verbose=True)
        heat=heat_print[0].toarray()
        diff=np.copy(heat)
        
        for i in range(len(A_tilde)):
            heat[i].sort()
        for i in range(len(A_tilde)):
            for j in range(len(A_tilde)):
                diff[i][j]=np.exp(-np.sum(abs(heat[i]-heat[j])) )

        
        for _ in range(self.order):
            A_tilde2=np.copy(A_tilde)
            A_tilde3=np.copy(A_tilde)
            for i in range(len(A_tilde2)):
                for j in range(len(A_tilde2[i])):
                    if(A_tilde2[i][j]>0):
                        A_tilde2[i][j]=graph.degree(j)
                        A_tilde3[i][j]=diff[i][j]
            A_tilde2=normalize(A_tilde2, axis=1, norm='l1')
            A_tilde3=normalize(A_tilde3, axis=1, norm='l1')

            X1 = A_tilde2.dot(X)
            X2 = A_tilde3.dot(X)
            feature_blocks.append(X1)
            feature_blocks.append(X2)
            A_tilde=A_tilde.dot(tmp)
        feature_blocks = np.concatenate(feature_blocks, axis=1)
        if self.pooling == "mean":
            feature_blocks = np.mean(feature_blocks, axis=0)
        elif self.pooling == "min":
            feature_blocks = np.min(feature_blocks, axis=0)
        elif self.pooling == "max":
            feature_blocks = np.max(feature_blocks, axis=0)
        else:
            raise ValueError("Wrong pooling function.")
        return feature_blocks


    def fit(self, graphs: List[nx.classes.graph.Graph]):

        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_WaveletC(graph) for graph in tqdm(graphs)]


    def get_embedding(self) -> np.array:

        return np.array(self._embedding)