import os
import subprocess
from datetime import datetime
from ConfigSpace import ConfigurationSpace, EqualsCondition, NotEqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    UnParametrizedHyperparameter,
)
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from search_space.utils import graph_to_matrix


class GRaSP:
    """Causal discovery with greedy relaxation of the sparsest permutation.

    References
    ----------
    Lam, Wai Yin and Andrews, Bryan and Ramsey, Joseph,
    "Greedy relaxations of the sparsest permutation algorithm", UAI 2022.

    Huang, Biwei and Zhang, Kun and Lin, Yizhu and Schölkopf, Bernhard and Glymour, Clark,
    "Generalized score functions for causal discovery", ACM SIGKDD international conference
    on knowledge discovery & data mining 2018.

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
        os.makedirs("causalcmd/", exist_ok=True)
        prefix = "causalcmd/grasp_" + str(datetime.now()).replace(" ", "_")
        if config["score"] == "bdeu-score":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "grasp",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--test", config["test"],
                "--score", config["score"],
                "--alpha", str(config["alpha"]),
                "--structurePrior", str(config["structure"]),
                "--numStarts", str(config["n_starts"]),
                "--graspDepth", str(config["depth"]),
                "--useDataOrder",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        elif config["score"] == "sem-bic-score" and config["test"] != "cci-test":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "grasp",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--test", config["test"],
                "--score", config["score"],
                "--alpha", str(config["alpha"]),
                "--penaltyDiscount", str(config["penalty"]),
                "--semBicStructurePrior", str(config["structure"]),
                "--numStarts", str(config["n_starts"]),
                "--graspDepth", str(config["depth"]),
                "--useDataOrder",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        elif config["score"] == "sem-bic-score" and config["test"] == "cci-test":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "grasp",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--test", config["test"],
                "--score", config["score"],
                "--alpha", str(config["alpha"]),
                "--kernelType", str(config["k_type"]),
                "--kernelMultiplier", str(config["k_multiplier"]),
                "--kernelRegressionSampleSize", str(config["k_sample_size"]),
                "--basisType", str(config["b_type"]),
                "--numBasisFunctions", str(config["b_num_func"]),
                "--penaltyDiscount", str(config["penalty"]),
                "--semBicStructurePrior", str(config["structure"]),
                "--numStarts", str(config["n_starts"]),
                "--graspDepth", str(config["depth"]),
                "--useDataOrder",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        elif config["test"] == "cci-test" and config["score"] != "sem-bic-score":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "grasp",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--test", config["test"],
                "--score", config["score"],
                "--alpha", str(config["alpha"]),
                "--kernelType", str(config["k_type"]),
                "--kernelMultiplier", str(config["k_multiplier"]),
                "--kernelRegressionSampleSize", str(config["k_sample_size"]),
                "--basisType", str(config["b_type"]),
                "--numBasisFunctions", str(config["b_num_func"]),
                "--penaltyDiscount", str(config["penalty"]),
                "--structurePrior", str(config["structure"]),
                "--numStarts", str(config["n_starts"]),
                "--graspDepth", str(config["depth"]),
                "--useDataOrder",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        elif self.data_type == "mixed":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "grasp",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--test", config["test"],
                "--score", config["score"],
                "--alpha", str(config["alpha"]),
                "--penaltyDiscount", str(config["penalty"]),
                "--structurePrior", str(config["structure"]),
                "--numStarts", str(config["n_starts"]),
                "--graspDepth", str(config["depth"]),
                "--numCategories", str(0), # required but isn't used
                "--useDataOrder",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        else:
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "grasp",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--test", config["test"],
                "--score", config["score"],
                "--alpha", str(config["alpha"]),
                "--penaltyDiscount", str(config["penalty"]),
                "--structurePrior", str(config["structure"]),
                "--numStarts", str(config["n_starts"]),
                "--graspDepth", str(config["depth"]),
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
            test = CategoricalHyperparameter(
                name="test",
                choices=["chi-square-test", "g-square-test", "cg-lr-test", "dg-lr-test"],
                default_value="chi-square-test"
            )
            score = CategoricalHyperparameter(
                name="score",
                choices=["bdeu-score", "disc-bic-score", "cg-bic-score", "dg-bic-score"],
                default_value="bdeu-score"
            )
            alpha = UniformFloatHyperparameter(
                name="alpha", lower=0.01, upper=0.05, default_value=0.05
            )
            penalty = UniformFloatHyperparameter(
                name="penalty", lower=0.0, upper=2.0, default_value=1.0
            )
            structure = UniformFloatHyperparameter(
                name="structure", lower=0.0, upper=2.0, default_value=0.0
            )
            n_starts = UniformIntegerHyperparameter(
                name="n_starts", lower=1, upper=10, default_value=5
            )
            depth = UniformIntegerHyperparameter(
                name="depth", lower=0, upper=10, default_value=1
            )
            config_space.add_hyperparameters(
                [test, score, alpha, penalty, structure, n_starts, depth]
            )
            cond1 = NotEqualsCondition(child=penalty, parent=score, value="bdeu-score")
            config_space.add_conditions([cond1])
        elif self.data_type == "continuous":
            test = CategoricalHyperparameter(
                name="test",
                choices=["fisher-z-test", "cci-test", "cg-lr-test", "dg-lr-test"],
                default_value="fisher-z-test"
            )
            score = CategoricalHyperparameter(
                name="score",
                choices=["sem-bic-score", "cg-bic-score", "dg-bic-score"],
                default_value="sem-bic-score"
            )
            alpha = UniformFloatHyperparameter(
                name="alpha", lower=0.01, upper=0.05, default_value=0.05
            )
            penalty = UniformFloatHyperparameter(
                name="penalty", lower=0.0, upper=2.0, default_value=1.0
            )
            structure = UniformFloatHyperparameter(
                name="structure", lower=0.0, upper=2.0, default_value=0.0
            )
            b_type = UnParametrizedHyperparameter(
                name="b_type", value=2
            )
            b_num_func = UnParametrizedHyperparameter(
                name="b_num_func", value=30
            )
            k_type = UnParametrizedHyperparameter(
                name="k_type", value=1
            )
            k_multiplier = UnParametrizedHyperparameter(
                name="k_multiplier", value=1
            )
            k_sample_size = UnParametrizedHyperparameter(
                name="k_sample_size", value=100
            )
            n_starts = UniformIntegerHyperparameter(
                name="n_starts", lower=1, upper=10, default_value=5
            )
            depth = UniformIntegerHyperparameter(
                name="depth", lower=0, upper=10, default_value=1
            )
            config_space.add_hyperparameters(
                [test, score, alpha, penalty, structure, b_type, b_num_func,
                 k_type, k_multiplier, k_sample_size, n_starts, depth]
            )
            cond1 = EqualsCondition(child=b_type, parent=test, value="cci-test")
            cond2 = EqualsCondition(child=b_num_func, parent=test, value="cci-test")
            cond3 = EqualsCondition(child=k_type, parent=test, value="cci-test")
            cond4 = EqualsCondition(child=k_multiplier, parent=test, value="cci-test")
            cond5 = EqualsCondition(child=k_sample_size, parent=test, value="cci-test")
            config_space.add_conditions([cond1, cond2, cond3, cond4, cond5])
        else:
            # mixed data
            test = CategoricalHyperparameter(
                name="test",
                choices=["cg-lr-test", "dg-lr-test"],
                default_value="cg-lr-test"
            )
            score = CategoricalHyperparameter(
                name="score",
                choices=["cg-bic-score", "dg-bic-score"],
                default_value="cg-bic-score"
            )
            alpha = UniformFloatHyperparameter(
                name="alpha", lower=0.01, upper=0.05, default_value=0.05
            )
            penalty = UniformFloatHyperparameter(
                name="penalty", lower=0.0, upper=2.0, default_value=1.0
            )
            structure = UniformFloatHyperparameter(
                name="structure", lower=0.0, upper=2.0, default_value=0.0
            )
            n_starts = UniformIntegerHyperparameter(
                name="n_starts", lower=1, upper=10, default_value=5
            )
            depth = UniformIntegerHyperparameter(
                name="depth", lower=0, upper=10, default_value=1
            )
            config_space.add_hyperparameters(
                [test, score, alpha, penalty, structure, n_starts, depth]
            )
        return config_space
