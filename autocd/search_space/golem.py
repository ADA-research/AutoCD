import numpy as np
import pandas as pd
from copy import deepcopy
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UnParametrizedHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from castle.algorithms import GOLEM as golem
from causallearn.utils.DAG2CPDAG import dag2cpdag
from search_space.utils import matrix_to_graph


class GOLEM:
    """Causal discovery using DAG-penalized likelihood to learn linear DAG models.

    References
    ----------
    Ng, Ignavier and Ghassami, AmirEmad and Zhang, Kun,
    "On the role of sparsity and dag constraints for learning linear dags", NeurIPS 2020.

    Implementation
    --------------
    Zhang, Keli and Zhu, Shengyu and Kalander, Marcus and Ng, Ignavier and
    Ye, Junjian and Chen, Zhitang and Pan, Lujia,
    "gcastle: A python toolbox for causal discovery", ArXiv 2021.

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
        data = pd.read_csv(datapath)
        params = deepcopy(config)
        params.update({"checkpoint_iter": int(params["num_iter"] * 0.1)})
        model = golem(**params)
        model.learn(data)
        dag = np.array(model.causal_matrix)
        cpdag = dag2cpdag(matrix_to_graph(dag))
        return dag, cpdag

    def get_cs(self):
        """Compute the search space given the data type.
        
        Returns
        ----------
        ConfigurationSpace object that contains search space for this algorithm.
        """
        config_space = ConfigurationSpace()
        # if self.data_type == "continuous":
        lambda_1 = UniformFloatHyperparameter(
            name="lambda_1", lower=0.01, upper=0.05, default_value=0.02
        )
        lambda_2 = UniformFloatHyperparameter(
            name="lambda_2", lower=1.0, upper=5.0, default_value=5.0
        )
        equal_variances = CategoricalHyperparameter(
            name="equal_variances", choices=[False], default_value=False
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.001, upper=0.005, default_value=0.001
        )
        num_iter = UniformIntegerHyperparameter(
            name="num_iter", lower=500, upper=2000, default_value=500
        )
        seed = UnParametrizedHyperparameter(name="seed", value=self.seed)
        graph_thres = UniformFloatHyperparameter(
            name="graph_thres", lower=0.1, upper=0.5, default_value=0.3
        )
        device_type = UnParametrizedHyperparameter(name="device_type", value="cpu")
        config_space.add_hyperparameters(
            [
                lambda_1,
                lambda_2,
                equal_variances,
                learning_rate,
                num_iter,
                seed,
                graph_thres,
                device_type,
            ]
        )
        return config_space
