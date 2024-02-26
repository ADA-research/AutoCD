import numpy as np
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode

def graph_to_matrix(matrix):
    """Transform the causal DAG adjacency matrix such that it only contains 0 and 1.
    Parameters
    ----------
    matrix: numpy.ndarray
        DAG adjacency matrix that needs to be transformed
    
    Returns
    -------
    Transformed causal DAG adjacency matrix.
    """
    n_features = len(matrix)
    transform = np.zeros((n_features, n_features), dtype=int)
    for i in range(n_features):
        for j in range(n_features):
            if matrix[i, j] == -1 and matrix[j, i] == 1:
                transform[i, j] = 1
                transform[j, i] = 0
            else:
                continue
    return transform


def matrix_to_graph(graph):
    """Transform the causal adjacency matrix to causal learn Graph object.
    Parameters
    ----------
    graph: numpy.ndarray
        DAG adjacency matrix
    
    Returns
    -------
    Tetrad Graph object of the given graph.
    """
    n_nodes = len(graph)
    nodes = []
    for i in range(n_nodes):
        nodes.append(GraphNode(f"X{i + 1}"))
    
    transform = Dag(nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if graph[i, j] == 1 and graph[j, i] == 0:
                transform.add_directed_edge(nodes[i], nodes[j])
            else:
                continue
    return transform