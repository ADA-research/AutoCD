import os
import subprocess
import pandas as pd
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UnParametrizedHyperparameter,
    UniformFloatHyperparameter,
)
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from search_space.utils import graph_to_matrix


class ICALiNGAM:
    """Causal discovery with ICA-based linear non-Gaussian acyclic model.

    References
    ----------
    Shimizu, Shohei and Hoyer, Patrik O. and Hyvärinen, Aapo and Kerminen, Antti and Jordan, Michael,
    "A linear non-Gaussian acyclic model for causal discovery", JMLR 2006.

    Implementation
    --------------
    Ramsey, Joseph D. and Zhang, Kun and Glymour, Madelyn and Romero, Ruben Sanchez and Huang, 
    Biwei and Ebert-Uphoff, Imme and Samarasinghe, Savini and Barnes, Elizabeth A. and Glymour, 
    Clark, "TETRAD - A toolbox for causal discovery", CI 2018.

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
        prefix = "causalcmd/ica_lingam_" + str(datetime.now()).replace(" ", "_")
        subprocess.run([
            "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
            "--algorithm", "ica-lingam",
            "--data-type", self.data_type,
            "--dataset", datapath,
            "--delimiter", "comma",
            "--fastIcaA", str(config["alpha"]),
            "--fastIcaTolerance", str(config["tolerance"]),
            "--fastIcaMaxIter", str(config["max_iter"]),
            "--guaranteeAcyclic",
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
            alpha = UniformFloatHyperparameter(
                name="alpha", lower=1.0, upper=2.0, default_value=1.1
            )
            max_iter = UniformFloatHyperparameter(
                name="max_iter", lower=2000.0, upper=5000.0, default_value=2000.0
            )
            tolerance = UniformFloatHyperparameter(
                name="tolerance", lower=1e-08, upper=1e-06, default_value=1e-06
            )
            b_thresh = UniformFloatHyperparameter(
                name="b_thresh", lower=0.1, upper=0.6, default_value=0.1
            )
            config_space.add_hyperparameters(
                [alpha, max_iter, tolerance, b_thresh]
            )
            return config_space
