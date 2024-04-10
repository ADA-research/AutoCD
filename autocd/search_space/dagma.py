import os
import subprocess
import pandas as pd
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from search_space.utils import graph_to_matrix


class DAGMA:
    """Causal discovery of dags via m-matrices and log-determinant acyclicity.

    References
    ----------
    Bello, Kevin and Aragam, Bryon and Ravikumar, Pradeep, 
    "Dagma: Learning dags via m-matrices and a log-determinant acyclicity characterization.", 
    NEURIPS 2022.
    
    Implementation
    --------------
    Ramsey, Joseph D. and Zhang, Kun and Glymour, Madelyn and Romero, Ruben Sanchez and Huang, 
    Biwei and Ebert-Uphoff, Imme and Samarasinghe, Savini and Barnes, Elizabeth A. and Glymour, Clark, 
    "TETRAD - A toolbox for causal discovery", CI 2018.

    Parameters
    ----------
    data_type: str
        data type of the dataset
    seed: int
        random seed
    """
    
    def __init__(self, data_type, seed):
        self.data_type = data_type
        self.seed = seed

    def estimate(self, datapath, config):
        """Build a causal discovery model and estimate the causal graph.
        
        Parameters
        ----------
        datapath: str
            path to the data
        config: dict
            set of hyperparameters to be configured

        Returns
        -------
        Adjacency matrix of the estimated causal graph in DAG and CPDAG.
        """
        try:
            pd.read_csv(datapath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file {datapath} was not found. Please make sure the file exists.")
            
        os.makedirs("causalcmd/", exist_ok=True)
        prefix = "causalcmd/dagma_" + str(datetime.now()).replace(" ", "_")
        subprocess.run([
            "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
            "--algorithm", "dagma",
            "--data-type", self.data_type,
            "--dataset", datapath,
            "--delimiter", "comma",
            "--lambda1", str(config["lambda"]),
            "--wThreshold", str(config["w_thresh"]),
            "--seed", str(self.seed),
            "--skip-validation",
            "--make-bidirected-undirected",
            "--prefix", prefix
        ])
        cpdag = txt2generalgraph(prefix + "_out.txt")
        dag = pdag2dag(cpdag).graph.astype(int)
        os.remove(prefix + "_out.txt")
        return graph_to_matrix(dag), cpdag

    def get_cs(self):
        """Compute the search space given the data type.
        
        Returns
        ----------
        ConfigurationSpace object that contains search space for this algorithm.
        """
        config_space = ConfigurationSpace()
        if self.data_type == "continuous":
            lambda1 = UniformFloatHyperparameter(
                name="lambda", lower=0.01, upper=0.05, default_value=0.05
            )
            w_thresh = UniformFloatHyperparameter(
                name="w_thresh", lower=0.1, upper=0.6, default_value=0.1
            )
            config_space.add_hyperparameters(
                [lambda1, w_thresh]
            )
            return config_space
