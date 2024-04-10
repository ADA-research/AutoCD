import jpype
import jpype.imports
try:
    jpype.startJVM(classpath=["pytetrad/tetrad-current.jar"])
except OSError:
    print("Unable to start JVM. Please make sure that the 'repo/autocd/pytetrad/tetrad-current.jar' file is present.")

import edu.cmu.tetrad.data as data
from edu.cmu.tetrad.util import Params, Parameters
from edu.cmu.tetrad.algcomparison.simulation import GeneralSemSimulation, BayesNetSimulation, LeeHastieSimulation
import edu.cmu.tetrad.algcomparison.graph.RandomForward as RandomForward

import os
import argparse
import numpy as np
import pandas as pd
from utils import save_json, load_json
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nodes", type=int, default=10, help="Number of nodes in the DAG (default 10)"
    )
    parser.add_argument(
        "--degree", type=int, default=3, help="Average node degree in the DAG (default 3)"
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=1000,
        help="Number of instances in the dataset (default is 1000)",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        nargs="+",
        choices=["continuous", "discrete", "mixed"],
        default=["continuous", "discrete", "mixed"],
        help="Data type of the graph models: continuous, discrete and/or mixed",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=25,
        help="Number of repetitions of the dataset (default is 25)",
    )
    parser.add_argument(
        "--subsamples", type=int, default=20, help="Number of subsamples for StARS"
    )
    parser.add_argument(
        "--folds", type=int, default=10, help="Number of folds for OCT"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed"
    )
    opts = parser.parse_args()
    return opts

def convert_tetrad_graph(graph):
    """Create adjacency matrix from Tetrad DAG.
    
    Parameters
    ----------
    graph: Tetrad Graph object
        directed acyclic graph (DAG)
    
    Returns
    -------
    Adjacency matrix of the given graph.
    """
    dag_map = {"ARROW": 1, "TAIL": 0, "NULL": 0, "CIRCLE": 0}
    n_nodes = graph.getNumNodes()
    nodes = graph.getNodes()
    matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    for edge in graph.getEdges():
        i = nodes.indexOf(edge.getNode1())
        j = nodes.indexOf(edge.getNode2())

        matrix[i, j] = dag_map[edge.getEndpoint2().name()]
        matrix[j, i] = dag_map[edge.getEndpoint1().name()]

    return matrix

def convert_tetrad_data(data):
    """Create pandas dataframe from Tetrad data object.
    
    Parameters
    ----------
    data: Tetrad DataModel object
        dataset
    
    Returns
    -------
    Dataframe of the dataset.
    """
    names = data.getVariableNames()
    columns_ = []

    for name in names:
        columns_.append(str(name))

    df = pd.DataFrame(columns=columns_, index=range(data.getNumRows()))

    for row in range(data.getNumRows()):
        for col in range(data.getNumColumns()):
            df.at[row, columns_[col]] = data.getObject(row, col)

    return df

def get_data_properties(dataframe):
    """Create data properties from the dataframe
    
    Parmeters
    ---------
    dataframe: pandas.DataFrame
        dataset
    
    Returns
    -------
    Data properties from the given dataset.
    """
    _, n_col = dataframe.shape
    is_cat = np.zeros(n_col, dtype=bool)
    domain = np.zeros(n_col, dtype=int)
    classes = [list()] * n_col
    for idx, col in enumerate(dataframe.columns):
        if np.floor(dataframe[col][0]) == dataframe[col][0]:
            is_cat[idx] = True
            classes[idx] = dataframe[col].unique().tolist()
            domain[idx] = max(classes[idx]) + 1
        else:
            classes[idx] = list()

    data_info = {
        "is_cat": is_cat.tolist(),
        "classes": classes,
        "domain": domain.tolist()
    }
    return data_info

def generate_data(opts, datatype, seed):
    """Generate graphical model and simulate data.
    
    Parameters
    ----------
    opts: kwargs
        information to generate graphical models and simulate data
    datatype: str
        the data type of the dataset
    seed: int
        random seed
    
    Returns
    -------
    Simulated dataset and the generated DAG.
    """
    min_c = 2
    max_c = 20
    perc_d = 50

    random_graph = RandomForward() # Erdos-Renyi, Scale-free, ...
    params = Parameters()
    params.set(Params.NUM_MEASURES, opts.nodes)
    params.set(Params.SAMPLE_SIZE, opts.instances)
    params.set(Params.AVG_DEGREE, opts.degree)
    params.set(Params.MAX_DEGREE, opts.degree + 1)
    params.set(Params.DIFFERENT_GRAPHS, True)
    params.set(Params.NUM_RUNS, 1)
    params.set(Params.NUM_LATENTS, 0) # Can be changed
    params.set(Params.RANDOMIZE_COLUMNS, False)
    params.set(Params.SEED, seed)

    if datatype == "continuous":
        sim = GeneralSemSimulation(random_graph)
    elif datatype == "discrete":
        params.set(Params.MIN_CATEGORIES, min_c)
        params.set(Params.MAX_CATEGORIES, max_c)
        sim = BayesNetSimulation(random_graph)
    elif datatype =="mixed":
        params.set(Params.MIN_CATEGORIES, min_c)
        params.set(Params.MAX_CATEGORIES, max_c)
        params.set(Params.PERCENT_DISCRETE, perc_d)
        sim = LeeHastieSimulation(random_graph)
    else:
        print("This data type does not exist")

    sim.createData(params, True)
    t_dag = sim.getTrueGraph(0)
    t_data = sim.getDataModel(0)

    # Save data and target graph
    data = convert_tetrad_data(t_data)
    dag = convert_tetrad_graph(t_dag)
    return data, dag

def create_splits(opts, datatype, seed):
    """Create data splits for SMAC using K-Fold cross-validation and subsampling.
    
    Parameters
    ----------
    opts: Namespace
        contains commandline arguments
    datatype: str
        the data type of the dataset
    seed: int    
        random seed
    """
    outputpath = f"splits/{datatype}_{opts.nodes}_{opts.degree}/seed-{seed}/"
    os.makedirs(outputpath, exist_ok=True)
    np.random.seed(seed)

    data = np.array(pd.read_csv(f"data/dataset/{datatype}_{opts.nodes}_{opts.degree}/dataset_{seed}.csv"))

    kfold = KFold(n_splits=opts.folds, shuffle=True)
    for i, (train_index, test_index) in enumerate(kfold.split(data)):
        train, valid = pd.DataFrame(data[train_index]), pd.DataFrame(data[test_index])
        train_file = f"train-{i}.csv"
        valid_file = f"valid-{i}.csv"
        train.to_csv(os.path.join(outputpath, train_file), index=False)
        valid.to_csv(os.path.join(outputpath, valid_file), index=False)

    n_samples, _ = data.shape
    n_sub = len(data) // 2
    for i in range(opts.subsamples):
        train = pd.DataFrame(data[np.random.choice(n_samples, size=n_sub, replace=False)])
        sample_file = f"sample-{i}.csv"
        train.to_csv(os.path.join(outputpath, sample_file), index=False)

def main(opts):
    for datatype in opts.data_type:
        os.makedirs(f"data/dataset/{datatype}_{opts.nodes}_{opts.degree}", exist_ok=True)
        os.makedirs(f"data/graph/{datatype}_{opts.nodes}_{opts.degree}", exist_ok=True)
        os.makedirs(f"data/info/{datatype}_{opts.nodes}_{opts.degree}", exist_ok=True)

        for run in range(opts.seed, opts.seed + opts.repetitions):
            data, dag = generate_data(opts, datatype, run)
            data.to_csv(f"data/dataset/{datatype}_{opts.nodes}_{opts.degree}/dataset_{run}.csv", index=False)
            save_json(dag.tolist(), f"data/graph/{datatype}_{opts.nodes}_{opts.degree}/graph_{run}")
            dataframe = pd.read_csv(f"data/dataset/{datatype}_{opts.nodes}_{opts.degree}/dataset_{run}.csv") # Need to be saved and reloaded
            info = get_data_properties(dataframe)
            save_json(info, f"data/info/{datatype}_{opts.nodes}_{opts.degree}/info_{run}")
            create_splits(opts, datatype, run)

def adni(opts):
    """Create dataset properties, target graph and data splits from the ADNI dataset
    Parameters
    ----------
    opts: Namespace
        contains commandline arguments
    """
    dataset = pd.read_csv("daßa/dataset/mixed_9_adni.csv")
    info = get_data_properties(dataset)

    # The "gold standard" graph according to Shen et al. (2020)
    graph[0, 4] = 1
    graph = np.array((dataset.shape[1], dataset.shape[1]), dtype=int)
    graph[2, 6] = 1
    graph[3, 6] = 1
    graph[4, 3] = 1
    graph[4, 5] = 1
    graph[5, 6] = 1
    graph[7, 4] = 1
    graph[8, 4] = 1
    save_json(info, "data/info/mixed_9_adni")
    save_json(graph.tolist(), "data/graph/mixed_9_adni")

    # Create data splits 
    data = np.array(dataset)
    for run in range(opts.seed, opts.seed + opts.repetitions):
        outputpath = f"splits/mixed_9_adni/seed-{run}/"
        os.makedirs(outputpath, exist_ok=True)
        np.random.seed(run)

        kfold = KFold(n_splits=opts.folds, shuffle=True)
        for i, (train_index, test_index) in enumerate(kfold.split(data)):
            train, valid = pd.DataFrame(data[train_index]), pd.DataFrame(data[test_index])
            train_file = f"train-{i}.csv"
            valid_file = f"valid-{i}.csv"
            train.to_csv(os.path.join(outputpath, train_file), index=False)
            valid.to_csv(os.path.join(outputpath, valid_file), index=False)

        n_samples, _ = data.shape
        n_sub = len(data) // 2
        for i in range(opts.subsamples):
            train = pd.DataFrame(data[np.random.choice(n_samples, size=n_sub, replace=False)])
            sample_file = f"sample-{i}.csv"
            train.to_csv(os.path.join(outputpath, sample_file), index=False)


if __name__ == "__main__":
    main(parse_args())
    # adni(parse_args())
