import numpy as np
import pandas as pd
import networkx as nx
import Relevance_CN


def graph(file_path='Power.csv'):
    file = pd.read_csv(file_path, low_memory=False)

    time = 'time'
    nodes = file.columns[file.columns != time].tolist()

    mi_matrix = Relevance_CN.mutual_information(file_path)

    G = nx.Graph()
    N = len(nodes)

    G.add_nodes_from(nodes)

    W = np.zeros((N, N))

    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    for i, node in enumerate(nodes):
        neighbors = pd.Series(mi_matrix[i], index=nodes)
        neighbors = neighbors[neighbors > 0.10]

        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
            j = node_to_index[neighbor]
            W[i, j] = weight
            W[j, i] = weight
    return G, W
