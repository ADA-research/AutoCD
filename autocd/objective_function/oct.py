import os
import numpy as np
import pandas as pd
from utils import load_pickle, save_pickle
from tetrad_utils import wrapper_cd
from objective_function.utils import (
    get_markov_boundary,
    empirical_prob,
    mutual_information,
)
from ConfigSpace import Configuration
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class OCT:
    """OCT algorithm as objective function for Bayesian optimization.

    Parameters
    ----------
    datapath: str
        path to the subsamples
    data_info: dict 
        data information
    autocd: boolean
        True if AutoCD is used and Talse otherwise
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
        self.n_folds = 10
        self.n_permutations = 1000
        self.threshold = 0.05

        self.train_data = [None] * self.n_folds
        self.valid_data = [None] * self.n_folds
        for fold in range(self.n_folds):
            self.train_data[fold] = f"{self.datapath}/train-{str(fold)}.csv"
            self.valid_data[fold] = f"{self.datapath}/valid-{str(fold)}.csv"

        self.pool_ys = [None] * self.n_features
        for node in range(self.n_features):
            pool_y = []
            for fold in range(self.n_folds):
                y_test = np.array(pd.read_csv(self.valid_data[fold]))[:, node]
                pool_y.extend(y_test)
            self.pool_ys[node] = pool_y
    
    def _estimate(self, config, graphs, using_smac):
        """Modified StARS method by Biza et al. (2022).

        Parameters
        ----------
        config: Configuration object
            a set of hyperparameters
        graphs: list
            contains the estimated graphs
        using_smac: boolean
            True if we are using SMAC and False if we use baseline

        Returns
        -------
        Average mutual information, average Markov boundary size and pool_yhats.
        """
        if using_smac:
            for fold in range(self.n_folds):
                print(f"Fold: {fold}")
                graphs[fold] = wrapper_cd(
                    self.train_data[fold], self.data_type, config, False, self.seed
                )

        mutual = np.zeros(self.n_features)
        mb_size = np.zeros(self.n_folds)
        pool_yhats = [None] * self.n_features
        for node in range(self.n_features):
            pool_yhat = []
            for fold in range(self.n_folds):
                if len(graphs[fold]) == 0:
                    return np.nan, np.nan, []
                else:
                    est_mbs = get_markov_boundary(graphs[fold])
                    mb_size[fold] = np.mean([len(mb) for mb in est_mbs])
                    mb = est_mbs[node]

                    y_train = np.array(pd.read_csv(self.train_data[fold]))[:, node]
                    x_train = np.array(pd.read_csv(self.train_data[fold]))[:, mb]
                    y_test = np.array(pd.read_csv(self.valid_data[fold]))[:, node]
                    x_test = np.array(pd.read_csv(self.valid_data[fold]))[:, mb]
                    n_test = y_test.shape[0]

                    if self.is_cat[node]:
                        x_train = x_train.astype("int")
                        y_train = y_train.astype("int")
                        x_test = x_test.astype("int")
                        if len(mb) > 0:
                            model = RandomForestClassifier(random_state=self.seed)
                            model.fit(x_train, y_train)
                            pred = model.predict(x_test)
                        else:
                            _, pred_val = empirical_prob(y_train)
                            scoreii = np.zeros(len(self.classes[node]), dtype="int")
                            for j in range(len(self.classes[node])):
                                scoreii[j] = np.count_nonzero(y_train == self.classes[node][j]) / len(
                                    y_train
                                )
                            pred = np.ones(n_test) * pred_val
                        pool_yhat.extend(pred.tolist())
                    else:
                        if len(mb) > 0:
                            model = RandomForestRegressor(random_state=self.seed)
                            model.fit(x_train, y_train)
                            pred = model.predict(x_test)
                        else:
                            pred = np.ones(n_test) * np.mean(y_train)
                        pool_yhat.extend(pred.tolist())

            pool_yhats[node] = np.array(pool_yhat)

            # Compute the predictive performance
            mutual[node] = mutual_information(
                self.is_cat, node, self.pool_ys[node], pool_yhats[node]
            )
        return 1 - np.mean(mutual), np.mean(mb_size), pool_yhats

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
        output_dir = f"output/oct/{self.dataname}/autocd_pc/run_{self.seed}/{self.seed}"
        mutual_info, mb_size, pool_yhats = self._estimate(config, [None] * self.n_folds, True)    
        
        if self.autocd:
            if not os.path.exists(f"{output_dir}/mb_size.pkl"):
                os.makedirs(output_dir, exist_ok=True)
                save_pickle([mb_size], f"{output_dir}/mb_size.pkl")
                save_pickle(self.pool_ys, f"{output_dir}/pool_ys.pkl")
                save_pickle([pool_yhats], f"{output_dir}/pool_yhats.pkl")
            else:
                exist_mb_size = load_pickle(f"{output_dir}/mb_size.pkl")
                exist_pool_yhats = load_pickle(f"{output_dir}/pool_yhats.pkl")
                exist_mb_size.append(mb_size)
                exist_pool_yhats.append(pool_yhats)
                save_pickle(exist_mb_size, f"{output_dir}/mb_size.pkl")
                save_pickle(exist_pool_yhats, f"{output_dir}/pool_yhats.pkl")
        
        return mutual_info

    def graphs(self, configuration, queue):
        """Compute the graphs for all folds for the given configuration.

        Parameters
        ----------
        configuration: dictionary
            contains information of the algorithm and its hyperparameters
        queue: Queue object
            multiprocess queue to store output
        """
        graphs = [None] * self.n_folds

        for fold in range(self.n_folds):
            print(f"Fold: {fold}")
            graphs[fold] = wrapper_cd(
                self.train_data[fold], self.data_type, configuration, False, self.seed
            )

        queue.put(graphs)

    def baseline(self, graphs, configs):
        """OCT algorithm with sparsity penalty by Biza et al. (2022).
        
        Parameters
        ----------
        graphs: list
            contains the estimated graph for each configuration and for each subsample
        configs: list
            contains all configurations
        
        Returns
        -------
        OCT value and configuration.
        """
        n_configs = len(graphs)
        mutual_info = np.full(n_configs, np.nan)
        mb_size = np.full(n_configs, np.nan)
        pool_yhats = [None] * n_configs
        for config in range(n_configs):
            if graphs[config] is not None:
                mutual_info[config], mb_size[config], pool_yhats[config] = self._estimate(configs[config], graphs[config], False)

        idxs = [None] * self.n_permutations
        is_equal = np.ones(n_configs)
        p_values = np.ones(n_configs)

        if np.all(np.isnan(mutual_info)):
            return np.nan, {}
        else:
            best_config = np.nanargmax(mutual_info)

            # Set idxs
            for i in range(self.n_permutations):
                idx = np.random.randint(0, 2, size=len(self.pool_ys[0]))
                idxs[i] = idx

            # Permutation testing
            for config in range(n_configs):
                if config == best_config or np.isnan(mutual_info[config]):
                    continue

                swap_best_metric = np.zeros((self.n_permutations, self.n_features))
                swap_cur_metric = np.zeros((self.n_permutations, self.n_features))

                for node in range(self.n_features):
                    pool_yhat_best = np.asarray(pool_yhats[best_config][node])
                    pool_yhat_cur = np.asarray(pool_yhats[config][node])
                    pool_y = np.asarray(self.pool_ys[node])

                    for i in range(self.n_permutations):
                        idx = np.asarray(idxs[i])

                        swap_best = np.copy(pool_yhat_best)
                        swap_best[idx] = pool_yhat_cur[idx]

                        swap_cur = np.copy(pool_yhat_cur)
                        swap_cur[idx] = pool_yhat_best[idx]

                        swap_best_metric[i, node] = mutual_information(
                            self.is_cat, node, pool_y, swap_best
                        )
                        swap_cur_metric[i, node] = mutual_information(
                            self.is_cat, node, pool_y, swap_cur
                        )

                cur_metric = mutual_info[config]
                best_metric = np.max(mutual_info)
                t_stat_obs = best_metric - cur_metric
                t_stat = np.mean(swap_best_metric, axis=1) - np.mean(
                    swap_cur_metric, axis=1
                )

                p_val = np.count_nonzero(t_stat >= t_stat_obs) / self.n_permutations
                p_values[config] = p_val

                # Hypothesis testing, H0: the difference in performance is zero
                is_equal[config] = 1 if p_val > self.threshold else 0

            # Select configuration
            config_penalty = best_config
            for config in range(n_configs):
                if np.isnan(mutual_info[config]):
                    continue

                if is_equal[config] and mb_size[config] < mb_size[config_penalty]:
                    config_penalty = config

            return mutual_info[config_penalty], configs[config_penalty]
