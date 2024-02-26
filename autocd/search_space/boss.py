import os
import subprocess
from datetime import datetime
from ConfigSpace import ConfigurationSpace, NotEqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from search_space.utils import graph_to_matrix


class BOSS:
    """Causal discovery with best order score search.

    References
    ----------
    Lam, Wai Yin and Andrews, Bryan and Ramsey, Joseph, 
    "Greedy relaxations of the sparsest permutation algorithm", PMLR 2022

    Teyssier, Marc and Koller, Daphne, 
    "Ordering-based search: A simple and effective algorithm for learning Bayesian networks.", 
    arXiv:1207.1429 2012.
    
    Solus, Liam and Wang, Yuhao and Uhler, Caroline, 
    "Consistency guarantees for greedy permutation-based causal inference algorithms." 
    Biometrika 2021.

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
        os.makedirs("causalcmd/", exist_ok=True)
        prefix = "causalcmd/boss_" + str(datetime.now()).replace(" ", "_")
        if config["score"] == "bdeu-score":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "boss",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--score", config["score"],
                "--structurePrior", str(config["structure"]),
                "--numStarts", str(config["n_start_thread"]),
                "--numThreads", str(config["n_start_thread"]),
                "--useBes",
                "--useDataOrder",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        elif config["score"] == "sem-bic-score":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "boss",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--score", config["score"],
                "--penaltyDiscount", str(config["penalty"]),
                "--semBicStructurePrior", str(config["structure"]),
                "--numStarts", str(config["n_start_thread"]),
                "--numThreads", str(config["n_start_thread"]),
                "--useBes",
                "--useDataOrder",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        elif self.data_type == "mixed":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "boss",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--score", config["score"],
                "--penaltyDiscount", str(config["penalty"]),
                "--structurePrior", str(config["structure"]),
                "--numStarts", str(config["n_start_thread"]),
                "--numThreads", str(config["n_start_thread"]),
                "--numCategories", str(0), # required but isn't used
                "--useBes",
                "--useDataOrder",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        else:
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "boss",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--score", config["score"],
                "--penaltyDiscount", str(config["penalty"]),
                "--structurePrior", str(config["structure"]),
                "--numStarts", str(config["n_start_thread"]),
                "--numThreads", str(config["n_start_thread"]),
                "--useBes",
                "--useDataOrder",
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
        if self.data_type == "discrete":
            score = CategoricalHyperparameter(
                name="score",
                choices=["bdeu-score", "disc-bic-score", "cg-bic-score", "dg-bic-score"],
                default_value="bdeu-score"
            )
            penalty = UniformFloatHyperparameter(
                name="penalty", lower=0.0, upper=2.0, default_value=1.0
            )
            structure = UniformFloatHyperparameter(
                name="structure", lower=0.0, upper=2.0, default_value=0.0
            )
            data_order = CategoricalHyperparameter(
                name="data_order", choices=["True", "False"], default_value="True"
            )
            n_start_thread = UniformIntegerHyperparameter(
                name="n_start_thread", lower=1, upper=10, default_value=5
            )
            config_space.add_hyperparameters(
                [score, penalty, structure, data_order, n_start_thread]
            )
            cond1 = NotEqualsCondition(child=penalty, parent=score, value="bdeu-score")
            config_space.add_conditions([cond1])
        elif self.data_type == "continuous":
            score = CategoricalHyperparameter(
                name="score",
                choices=["sem-bic-score", "cg-bic-score", "dg-bic-score"],
                default_value="sem-bic-score"
            )
            penalty = UniformFloatHyperparameter(
                name="penalty", lower=0.0, upper=2.0, default_value=1.0
            )
            structure = UniformFloatHyperparameter(
                name="structure", lower=0.0, upper=2.0, default_value=0.0
            )
            data_order = CategoricalHyperparameter(
                name="data_order", choices=["True", "False"], default_value="True"
            )
            n_start_thread = UniformIntegerHyperparameter(
                name="n_start_thread", lower=1, upper=10, default_value=5
            )
            config_space.add_hyperparameters(
                [score, penalty, structure, data_order, n_start_thread]
            )
        else:
            # mixed data
            score = CategoricalHyperparameter(
                name="score",
                choices=["cg-bic-score", "dg-bic-score"],
                default_value="cg-bic-score"
            )
            penalty = UniformFloatHyperparameter(
                name="penalty", lower=0.0, upper=2.0, default_value=1.0
            )
            structure = UniformFloatHyperparameter(
                name="structure", lower=0.0, upper=2.0, default_value=0.0
            )
            data_order = CategoricalHyperparameter(
                name="data_order", choices=["True", "False"], default_value="True"
            )
            n_start_thread = UniformIntegerHyperparameter(
                name="n_start_thread", lower=1, upper=10, default_value=5
            )
            config_space.add_hyperparameters(
                [score, penalty, structure, data_order, n_start_thread]
            )
        return config_space
