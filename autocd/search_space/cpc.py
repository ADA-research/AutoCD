import os
import subprocess
from datetime import datetime
from ConfigSpace import ConfigurationSpace, EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UnParametrizedHyperparameter,
    UniformFloatHyperparameter,
)
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from search_space.utils import graph_to_matrix


class CPC:
    """Causal discovery with convervative PC.

    References
    ----------
    Ramsey, Joseph and Spirtes, Peter and Zhang, Jiji, 
    "Adjacency-faithfulness and conservative causal inference.", UAI 2006.

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
        prefix = "causalcmd/cpc_" + str(datetime.now()).replace(" ", "_")
        if config["test"] == "cci-test":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "cpc",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--conflictRule", str(config["rule"]),
                "--test", config["test"],
                "--alpha", str(config["alpha"]),
                "--kernelType", str(config["k_type"]),
                "--kernelMultiplier", str(config["k_multiplier"]),
                "--kernelRegressionSampleSize", str(config["k_sample_size"]),
                "--basisType", str(config["b_type"]),
                "--numBasisFunctions", str(config["b_num_func"]),
                "--depth", str(config["depth"]),
                "--meekPreventCycles",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        elif self.data_type == "mixed":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "cpc",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--conflictRule", str(config["rule"]),
                "--test", config["test"],
                "--alpha", str(config["alpha"]),
                "--depth", str(config["depth"]),
                "--numCategories", str(0), # required but isn't used
                "--meekPreventCycles",
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        else:
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "cpc",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--conflictRule", str(config["rule"]),
                "--test", config["test"],
                "--alpha", str(config["alpha"]),
                "--depth", str(config["depth"]),
                "--meekPreventCycles",
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
            alpha = UniformFloatHyperparameter(
                name="alpha", lower=0.01, upper=0.05, default_value=0.05
            )
            test = CategoricalHyperparameter(
                name="test",
                choices=["chi-square-test", "g-square-test", "cg-lr-test", "dg-lr-test"],
                default_value="chi-square-test"
            )
            rule = CategoricalHyperparameter(
                name="rule",
                choices=[1, 2, 3],
                default_value=1
            )
            depth = UnParametrizedHyperparameter(
                name="depth", value=-1
            )
            config_space.add_hyperparameters(
                [alpha, test, rule, depth]
            )
        elif self.data_type == "continuous":
            alpha = UniformFloatHyperparameter(
                name="alpha", lower=0.01, upper=0.05, default_value=0.05
            )
            test = CategoricalHyperparameter(
                name="test",
                choices=["fisher-z-test", "cci-test", "cg-lr-test", "dg-lr-test"],
                default_value="fisher-z-test"
            )
            rule = CategoricalHyperparameter(
                name="rule",
                choices=[1, 2, 3],
                default_value=1
            )
            depth = UnParametrizedHyperparameter(
                name="depth", value=-1
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
            config_space.add_hyperparameters(
                [alpha, test, rule, depth, b_type, b_num_func, k_type, k_multiplier, k_sample_size]
            )
            cond1 = EqualsCondition(child=b_type, parent=test, value="cci-test")
            cond2 = EqualsCondition(child=b_num_func, parent=test, value="cci-test")
            cond3 = EqualsCondition(child=k_type, parent=test, value="cci-test")
            cond4 = EqualsCondition(child=k_multiplier, parent=test, value="cci-test")
            cond5 = EqualsCondition(child=k_sample_size, parent=test, value="cci-test")
            config_space.add_conditions([cond1, cond2, cond3, cond4, cond5])
        else:
            # mixed data
            alpha = UniformFloatHyperparameter(
                name="alpha", lower=0.01, upper=0.05, default_value=0.05
            )
            test = CategoricalHyperparameter(
                name="test",
                choices=["cg-lr-test", "dg-lr-test"],
                default_value="cg-lr-test"
            )
            rule = CategoricalHyperparameter(
                name="rule",
                choices=[1, 2, 3],
                default_value=1
            )
            depth = UnParametrizedHyperparameter(
                name="depth", value=-1
            )
            config_space.add_hyperparameters(
                [alpha, test, rule, depth]
            )
        return config_space
