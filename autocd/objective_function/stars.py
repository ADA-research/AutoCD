import os
import numpy as np
from copy import deepcopy
from ConfigSpace import Configuration
from utils import save_pickle, load_pickle
from tetrad_utils import wrapper_cd


class StARS:
    """StARS algorithm as objective function for Bayesian optimization.
    
    Parameters
    ----------
    datapath: str
        path to the subsamples
    data_info: dict 
        data information
    autocd: boolean
        True if AutoCD is used and False otherwise
    seed: int
        random seed
    """
    def __init__(self, datapath, data_info, autocd, seed):
        self.seed = seed
        self.autocd = autocd
        self.datapath = datapath
        self.dataname = self.datapath.split("/")[1]
        self.data_type = self.dataname.split("_")[0]
        self.n_features = int(self.dataname.split("_")[1])

        self.data_info = data_info
        self.is_cat = self.data_info["is_cat"]
        self.classes = self.data_info["classes"]
        self.domain = self.data_info["domain"]

        # Hyperparameters set by Biza et al. (2022)
        self.n_sets = 20
        self.threshold = 0.05

        self.subsamples = [None] * self.n_sets
        for i in range(self.n_sets):
            self.subsamples[i] = f"{self.datapath}/sample-{str(i)}.csv"

        self.n_edges_complete = self.n_features * (self.n_features - 1) // 2

    def _estimate(self, config, graphs):
        """Modified OCT method by Biza et al. (2022).

        Parameters
        ----------
        config: Configuration object
            a set of hyperparameters
        graphs: list
            contains the estimated graphs

        Returns
        -------
        Average sparsity value and instability value of the given configuration.
        """
        sparsity = np.zeros(self.n_sets)
        count = np.zeros((self.n_features, self.n_features))
        graph_empty = False
        for sets in range(self.n_sets):
            if graphs:
                graph = graphs[sets]
            else:
                graph = wrapper_cd(
                    self.subsamples[sets], self.data_type, config, False, self.seed
                )
            sparsity[sets] = np.count_nonzero(graph)

            if len(graph) == 0:
                return np.nan, np.nan
            else:
                for X in range(self.n_features):
                    for Y in range(X + 1, self.n_features):
                        if graph[X][Y] != 0 or graph[Y][X] != 0:
                            count[X][Y] += 1
        
        p_xy = count / self.n_sets
        indv_instability = 2 * p_xy * (1 - p_xy)
        instability = np.sum(indv_instability) / self.n_edges_complete
        return np.mean(sparsity), instability

    def tuning(self, config: Configuration, seed: int) -> float:
        """Loss function for Bayesian optimization.

        Parameters
        ----------
        config: Configuration object
            a set of hyperparameters
        seed: int
            random seed
        
        Returns
        -------
        Float that indicates performance of the configuration (lower is better).
        """
        output_dir = f"output/stars/{self.dataname}/autocd_pc/run_{self.seed}/{self.seed}"
        sparsity, instability = self._estimate(config, [])

        if self.autocd:
            if not os.path.exists(f"{output_dir}/sparsity.pkl"):
                os.makedirs(output_dir, exist_ok=True)
                save_pickle([sparsity], f"{output_dir}/sparsity.pkl")
            else:
                existing_list = load_pickle(f"{output_dir}/sparsity.pkl")
                existing_list.append(sparsity)
                save_pickle(existing_list, f"{output_dir}/sparsity.pkl")
        
        return instability

    def graphs(self, configuration, queue):
        """Compute the graphs for all subsamples for the given configuration.

        Parameters
        ----------
        configuration: dictionary
            contains information of the algorithm and its hyperparameters
        queue: Queue object
            multiprocess queue to store output
        """
        graphs = [None] * self.n_sets

        for sets in range(self.n_sets):
            print(f"Set: {sets}")
            graphs[sets] = wrapper_cd(
                self.subsamples[sets], self.data_type, configuration, False, self.seed
            )

        queue.put(graphs)

    def baseline(self, graphs, configs):
        """StARS algorithm with sparsity penalty by Biza et al. (2022).
        
        Parameters
        ----------
        graphs: list
            contains the estimated graph for each configuration and for each subsample
        configs: list
            contains all configurations
        
        Returns
        -------
        StARS value and configuration.
        """
        n_configs = len(configs)
        sparsity = np.full(n_configs, np.nan)
        instability = np.full(n_configs, np.nan)
        for config in range(n_configs):
            if graphs[config]:
                sparsity[config], instability[config] = self._estimate(configs[config], graphs[config])

        # Sort instability based on sparsity
        sort_idx = np.argsort(sparsity)
        sort_instability = instability[sort_idx]

        # Monotonize instability
        max_instability = deepcopy(sort_instability)
        for i in range(1, len(sort_instability)):
            max_instability[i] = max(max_instability[i], max_instability[i - 1])

        # Select configuration
        idx = np.where(max_instability <= self.threshold)[0]
        if len(idx) != 0:
            opt_idx = idx[-1]
        else:
            opt_idx = 0

        value = max_instability[opt_idx]
        config = configs[sort_idx[opt_idx]]

        if np.isnan(value):
            config = {}

        return value, config
