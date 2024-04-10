import os
import json
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from queue import Empty
from copy import deepcopy
from ConfigSpace import Configuration
from objective_function.utils import (
    get_markov_boundary,
    empirical_prob,
    mutual_information,
)
from objective_function.stars import StARS
from objective_function.oct import OCT
from tetrad_utils import wrapper_cd, configuration_space
from utils import load_json, load_pickle, save_pickle
from search_space.utils import matrix_to_graph
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True, help="Directory to the datasets"
    )
    parser.add_argument(
        "--result_dir", required=True, help="Directory to the configurator runs"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5,
        help="Sample size in the boostrapping procedure"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples in the bootstrapping procedure"
    )
    parser.add_argument(
        "--trial_walltime_limit",
        type=int,
        default=900,
        help=""
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=25,
        help="Number of configurator runs"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    opts = parser.parse_args()
    return opts

def wrapper_queue(datapath, data_type, config, cpdag, run, queue):
    """Wrapper for causalcmd to include termination.
    
    Parameters
    ----------
    datapath: str
        path to the dataset
    data_type: str
        the data type of the dataset
    config: dictionary
        contains information of the algorithms and its hyperparameters
    cpdag: boolean
        True if it should output a CPDAG and False if it should output a DAG.
    run: int
        random seed
    queue: Queue object
        mutliprocess queue to store output
    """
    graph = wrapper_cd(datapath, data_type, config, cpdag, run)
    queue.put(graph)

def get_cpdag(dag):
    """Compute the CPDAG of the given DAG.
    
    Parameters
    ----------
    dag: numpy.ndarray
        adjacency matrix of the dag
    
    Returns
    -------
    CPDAG as a causal learn Graph object.
    """
    dag_object = matrix_to_graph(dag)
    cpdag_object = dag2cpdag(dag_object)
    return cpdag_object

def evaluate(true_graph, est_graph):
    """Compute evaluation metrics.
    
    Parameters
    ----------
    true_graph: causal learn Graph object
        the target graph
    est_graph: causal learn Graph object
        the estimated graph
    
    Returns
    -------
    Normalized SHD, CA, FP, FN values.
    """
    edge_conf_matrix = AdjacencyConfusion(true_graph, est_graph)
    
    edge_tp = edge_conf_matrix.get_adj_tp()
    edge_tn = edge_conf_matrix.get_adj_tn()
    edge_fp = edge_conf_matrix.get_adj_fp()
    edge_fn = edge_conf_matrix.get_adj_fn()

    n_edges = (edge_tp + edge_fp + edge_fn)
    accuracy = edge_tp / n_edges
    shd = SHD(true_graph, est_graph).get_shd() / n_edges
    fp = edge_fp / n_edges
    fn = edge_fn / n_edges
    return shd, accuracy, fp, fn

def bootstrapping(configurations, sample_size, n_samples):
    """Create bootstrap distribution of the results.
    
    Parameters
    ----------
    configurations: list
        contains configurations
    sample_size: int
        the sample size
    n_samples: int
        the number the samples

    Returns
    -------
    Dataframe containing the bootstrap distribution.
    """
    bootstrap_configs = []
    for _ in range(n_samples):
        idxs = np.random.choice(configurations.index, sample_size, replace=False)
        samples = configurations.loc[idxs].dropna()
        if len(samples) == 0:
            bootstrap_configs.append((np.nan, np.nan, np.nan, np.nan, np.nan, ""))
        else:
            best = samples.loc[samples.iloc[:, 0].dropna().idxmin()]
            bootstrap_configs.append(best.to_list())

    dataframe = pd.DataFrame(bootstrap_configs, columns=configurations.columns)
    return dataframe

def stars_penalty(instability, sparsities, configs):
    """The StARS penalty by Liu et al. (2010) and Biza et al. (2022).
    
    Parameters
    ----------
    instability: list
        contains loss values from SMAC output
    sparsities: list
        contains the average number of edges from SMAC output
    configs: list
        contains the configurations from SMAC output
    
    Returns
    -------
    Best loss value according to penalty and corresponding configuration and sparsity value.
    """
    threshold = 0.05    # Hyperparameter set by Biza et al. (2022)
    sort_idx = np.argsort(sparsities)
    sort_instability = instability[sort_idx]
    sort_configs = configs[sort_idx]

    # Monotonize instability
    max_instability = deepcopy(sort_instability)
    for i in range(1, len(sort_instability)):
        max_instability[i] = max(max_instability[i], max_instability[i - 1])

    # Select configuration
    idx = np.where(max_instability <= threshold)[0]
    if len(idx) != 0:
        opt_idx = idx[-1]
    else:
        opt_idx = 0

    return sort_instability[opt_idx], sort_configs[opt_idx], sort_idx[opt_idx]

def oct_penalty(mutual_info, mb_size, pool_ys, pool_yhats, is_cat, configs):
    """The OCT penalty by Biza et al. (2022).
    
    Parameters
    ----------
    mutual_info: list
        contains the loss values from SMAC output
    mb_size: list
        contains the average Markov boundary size from SMAC output
    pool_ys: list
        contains the true variables
    pool_yhats: list
        contains the predicted variables
    is_cat: boolean
        True if variables is categorical/discrete and False otherwise
    configs: list
        contains the configurations from SMAC output
    
    Returns
    -------
    Best loss value according to the penalty and corresponding configuration and index.
    """
    threshold = 0.05
    n_permutations = 1000
    n_features = len(is_cat)
    n_configs = len(mutual_info)
    idxs = [None] * n_permutations
    is_equal = np.ones(n_configs)
    p_values = np.ones(n_configs)

    if np.all(np.isnan(mutual_info)):
        return np.nan, "", np.nan
    else:
        best_config = np.nanargmax(mutual_info)

        # Set idxs
        for i in range(n_permutations):
            idx = np.random.randint(0, 2, size=len(pool_ys[0]))
            idxs[i] = idx

        # Permutation testing
        for config in range(n_configs):
            print(f"Config: {config}/{n_configs - 1}")
            if (config == best_config) or (np.isnan(mutual_info[config])) or (not pool_yhats[config]):
                continue

            swap_best_metric = np.zeros((n_permutations, n_features))
            swap_cur_metric = np.zeros((n_permutations, n_features))

            for node in range(n_features):
                pool_yhat_best = np.asarray(pool_yhats[best_config][node])
                pool_yhat_cur = np.asarray(pool_yhats[config][node])
                pool_y = np.asarray(pool_ys[node])

                for i in range(n_permutations):
                    idx = np.asarray(idxs[i])

                    swap_best = np.copy(pool_yhat_best)
                    swap_best[idx] = pool_yhat_cur[idx]

                    swap_cur = np.copy(pool_yhat_cur)
                    swap_cur[idx] = pool_yhat_best[idx]

                    swap_best_metric[i, node] = mutual_information(
                        is_cat, node, pool_y, swap_best
                    )
                    swap_cur_metric[i, node] = mutual_information(
                        is_cat, node, pool_y, swap_cur
                    )

            cur_metric = mutual_info[config]
            best_metric = np.max(mutual_info)
            t_stat_obs = best_metric - cur_metric
            t_stat = np.mean(swap_best_metric, axis=1) - np.mean(
                swap_cur_metric, axis=1
            )

            p_val = np.count_nonzero(t_stat >= t_stat_obs) / n_permutations
            p_values[config] = p_val

            # Hypothesis testing, H0: the difference in performance is zero
            is_equal[config] = 1 if p_val > threshold else 0

        # Select configuration
        config_penalty = best_config
        for config in range(n_configs):
            if np.isnan(mutual_info[config]):
                continue

            if is_equal[config] and mb_size[config] < mb_size[config_penalty]:
                config_penalty = config

        return 1 - mutual_info[config_penalty], configs[config_penalty], config_penalty

def run_stars(opts):
    """Run the StARS penalty, store the results, and do bootstrapping.
    
    Parameters
    ----------
    opts: Namespace
        contains commandline arguments
    """
    # data_dir = "data/dataset/discrete_10_3"
    # result_dir = "output/stars/discrete_10_3/autocd"
    output_configurators = []
    dataset = opts.data_dir.split("/")[-1]
    objective = opts.result_dir.split("/")[1]
    method = opts.result_dir.split("/")[-1]
    
    for run in range(opts.seed, opts.seed + opts.repetitions):
        print(f"Run: {run}")
        if dataset == "mixed_9_shen":
            true_graph = np.array(load_json(f"data/graph/{dataset}"))
        else:
            true_graph = np.array(load_json(f"data/graph/{dataset}/graph_{run}"))
        
        est_graphs = load_pickle(f"{opts.result_dir}/run_{run}/{run}/graphs.pkl")
        sparsities = np.array(load_pickle(f"{opts.result_dir}/run_{run}/{run}/sparsity.pkl"))
        runhistory = load_json(f"{opts.result_dir}/run_{run}/{run}/runhistory.json")["data"]
        configs = load_json(f"{opts.result_dir}/run_{run}/{run}/runhistory.json")["configs"]
        intensifier = load_json(f"{opts.result_dir}/run_{run}/{run}/intensifier.json")
        incumbent = intensifier["trajectory"][-1]
        best_idx = incumbent["config_ids"][0] - 1
        best_value = incumbent["costs"][0]
        best_config = configs[str(best_idx + 1)]
        configs = np.array([value for value in configs.values()])

        instabilities = np.zeros(len(runhistory))
        for i in range(len(instabilities)):
            if runhistory[i][4] == np.inf or len(est_graphs[i]) == 0:
                instabilities[i] = np.nan
            else:
                instabilities[i] = runhistory[i][4]

        if not np.sum(~np.isnan(instabilities)) == 1:
            best_value, best_config, best_idx = stars_penalty(instabilities, sparsities, configs)
        
        if np.isnan(best_idx) or np.isnan(best_value):
            output_configurators.append(
                (np.nan, np.nan, np.nan, np.nan, np.nan, "")
            )
        else:
            true_graph = get_cpdag(true_graph)
            est_graph = get_cpdag(est_graphs[best_idx])
            shd, accuracy, fp, fn = evaluate(true_graph, est_graph)
            output_configurators.append(
                (best_value, shd, accuracy, fp, fn, best_config)
            )

    dataframe = pd.DataFrame(
        output_configurators,
        columns=[f"{objective}", "shd", "accuracy", "fp", "fn", "config"]
    )
    dataframe["config"] = dataframe["config"].apply(json.dumps)
    os.makedirs(f"results/{objective}/{dataset}", exist_ok=True)
    dataframe.to_csv(f"results/{objective}/{dataset}/{method}+.csv", index=False)

    results = bootstrapping(dataframe, opts.sample_size, opts.n_samples)
    os.makedirs(f"bootstrap/{objective}/{dataset}", exist_ok=True)
    results.to_csv(f"bootstrap/{objective}/{dataset}/{method}+.csv", index=False)

def run_oct(opts):
    """Run the OCT penalty, store the results, and do bootstrapping.
    
    Parameters
    ----------
    opts: Namespace
        contains commandline arguments
    """
    # data_dir = "data/dataset/discrete_10_3"
    # result_dir = "output/oct/discrete_10_3/autocd"
    output_configurators = []
    dataset = opts.data_dir.split("/")[-1]
    objective = opts.result_dir.split("/")[1]
    method = opts.result_dir.split("/")[-1]

    for run in range(opts.seed, opts.seed + opts.repetitions):
        print(f"Run: {run}")
        if dataset == "mixed_9_shen":
            true_graph = np.array(load_json(f"data/graph/{dataset}"))
            is_cat = load_json(f"data/info/{dataset}")["is_cat"]
        else:
            true_graph = np.array(load_json(f"data/graph/{dataset}/graph_{run}"))
            is_cat = load_json(f"data/info/{dataset}/info_{run}")["is_cat"]
        
        est_graphs = load_pickle(f"{opts.result_dir}/run_{run}/{run}/graphs.pkl")
        mb_size = np.array(load_pickle(f"{opts.result_dir}/run_{run}/{run}/mb_size.pkl"))
        pool_ys = load_pickle(f"{opts.result_dir}/run_{run}/{run}/pool_ys.pkl")
        pool_yhats = load_pickle(f"{opts.result_dir}/run_{run}/{run}/pool_yhats.pkl")
        runhistory = load_json(f"{opts.result_dir}/run_{run}/{run}/runhistory.json")["data"]
        configs = load_json(f"{opts.result_dir}/run_{run}/{run}/runhistory.json")["configs"]
        intensifier = load_json(f"{opts.result_dir}/run_{run}/{run}/intensifier.json")
        incumbent = intensifier["trajectory"][-1]
        best_idx = incumbent["config_ids"][0] - 1
        best_value = incumbent["costs"][0]
        best_config = configs[str(best_idx + 1)]
        configs = np.array([value for value in configs.values()])
        
        mutual_info = np.zeros(len(runhistory))
        for i in range(len(mutual_info)):
            if runhistory[i][4] == np.inf or len(est_graphs[i]) == 0:
                mutual_info[i] = np.nan
            else:
                mutual_info[i] = 1 - runhistory[i][4]

        if not np.sum(~np.isnan(mutual_info)) == 1:
            best_value, best_config, best_idx = oct_penalty(mutual_info, mb_size, pool_ys, pool_yhats, is_cat, configs)

        if np.isnan(best_idx) or np.isnan(best_value):
            output_configurators.append(
                (np.nan, np.nan, np.nan, np.nan, np.nan, "")
            )
        else:
            true_graph = get_cpdag(true_graph)
            est_graph = get_cpdag(est_graphs[best_idx])
            shd, accuracy, fp, fn = evaluate(true_graph, est_graph)
            output_configurators.append(
                (best_value, shd, accuracy, fp, fn, best_config)
            )

    dataframe = pd.DataFrame(
        output_configurators,
        columns=[f"{objective}", "shd", "accuracy", "fp", "fn", "config"]
    )
    dataframe["config"] = dataframe["config"].apply(json.dumps)
    os.makedirs(f"results/{objective}/{dataset}", exist_ok=True)
    dataframe.to_csv(f"results/{objective}/{dataset}/{method}+.csv", index=False)

    results = bootstrapping(dataframe, opts.sample_size, opts.n_samples)
    os.makedirs(f"bootstrap/{objective}/{dataset}", exist_ok=True)
    results.to_csv(f"bootstrap/{objective}/{dataset}/{method}+.csv", index=False)

def run_eval(opts):
    """Run the evaluation, store the results, and do bootstrapping.
    
    Parameters
    ----------
    opts: Namespace
        contains commandline arguments
    """
    # data_dir = "data/dataset/discrete_10_3"
    # result_dir = "output/stars/discrete_10_3/baseline"
    output_configurators = []
    dataset = opts.data_dir.split("/")[-1]
    data_type = dataset.split("_")[0]
    method = opts.result_dir.split("/")[-1]
    objective = opts.result_dir.split("/")[1]
    config_space = configuration_space(data_type, "autocd", opts.seed)
    
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for run in range(opts.seed, opts.seed + opts.repetitions):
        print(f"Run: {run}")
        if method.startswith("autocd"):
            est_graphs = load_pickle(f"{opts.result_dir}/run_{run}/{run}/graphs.pkl")
            intensifier = f"{opts.result_dir}/run_{run}/{run}/intensifier.json"
            runhistory = f"{opts.result_dir}/run_{run}/{run}/runhistory.json"
            configs = load_json(runhistory)["configs"]
            incumbent = load_json(intensifier)["trajectory"][-1]
            best_idx = incumbent["config_ids"][0] - 1
            best_value = incumbent["costs"][0]
            best_config = configs[str(best_idx + 1)]
        elif method == "baseline":
            dataframe = pd.read_csv(f"{opts.result_dir}.csv")
            best_value = dataframe["Value"][run]
            best_config = json.loads(dataframe["Configuration"][run])
        else:
            intensifier = f"{opts.result_dir}/run_{run}/{run}/intensifier.json"
            runhistory = f"{opts.result_dir}/run_{run}/{run}/runhistory.json"
            configs = load_json(runhistory)["configs"]
            incumbent = load_json(intensifier)["trajectory"][-1]
            best_idx = incumbent["config_ids"][0] - 1
            best_value = incumbent["costs"][0]
            best_config = configs[str(best_idx + 1)]

        if dataset == "mixed_9_shen":
            datapath = f"{opts.data_dir}.csv"
            true_graph = np.array(load_json(f"data/graph/{dataset}"))
        else:
            datapath = f"{opts.data_dir}/dataset_{run}.csv"
            true_graph = np.array(load_json(f"data/graph/{dataset}/graph_{run}"))
        
        if np.isinf(best_value) or not best_config:
            print("Encountered Infinity value")
            output_configurators.append((np.nan, np.nan, np.nan, np.nan, np.nan, ""))
        else:
            configuration = Configuration(config_space, best_config)
            if method.startswith("autocd"):
                if len(est_graphs[best_idx]) == 0:
                    incumbent = load_json(intensifier)["trajectory"][-2]
                    best_idx = incumbent["config_ids"][0] - 1
                    best_value = incumbent["costs"][0]
                    best_config = configs[str(best_idx + 1)]

                true_graph = get_cpdag(true_graph)
                est_graph = get_cpdag(est_graphs[best_idx])
                shd, accuracy, fp, fn = evaluate(true_graph, est_graph)
                output_configurators.append(
                    (best_value, shd, accuracy, fp, fn, best_config)
                )
            else:
                process = ctx.Process(
                    target=wrapper_queue, 
                    args=(datapath, data_type, configuration, True, run, queue), 
                    daemon=True
                )
                process.start()
                try:
                    est_graph = queue.get(True, opts.trial_walltime_limit)
                    true_graph = get_cpdag(true_graph)
                    shd, accuracy, fp, fn = evaluate(true_graph, est_graph)
                    output_configurators.append(
                        (best_value, shd, accuracy, fp, fn, best_config)
                    )
                except Empty:
                    process.kill()
                    print("Encountered Infinity value")
                    output_configurators.append((np.nan, np.nan, np.nan, np.nan, np.nan, ""))

    dataframe = pd.DataFrame(
        output_configurators,
        columns=[f"{objective}", "shd", "accuracy", "fp", "fn", "config"]
    )
    dataframe["config"] = dataframe["config"].apply(json.dumps)
    os.makedirs(f"results/{objective}/{dataset}", exist_ok=True)
    dataframe.to_csv(f"results/{objective}/{dataset}/{method}.csv", index=False)

    results = bootstrapping(dataframe, opts.sample_size, opts.n_samples)
    os.makedirs(f"bootstrap/{objective}/{dataset}", exist_ok=True)
    results.to_csv(f"bootstrap/{objective}/{dataset}/{method}.csv", index=False)

def compute_graphs(opts):
    """Compute graphs for each SMAC run for easier evaluation.
    
    Parameters
    ----------
    opts: Namespace
        contains commandline arguments
    """
    # result_dir = output/objective/discrete_10_3/method
    dataset = opts.result_dir.split("/")[2]
    data_type = dataset.split("_")[0]
    config_space = configuration_space(data_type, "autocd", opts.seed)
    for run in range(opts.seed, opts.seed + opts.repetitions):
        print(f"Run: {run}")
        datapath = f"data/dataset/{dataset}/dataset_{run}.csv"
        # datapath = f"data/dataset/{dataset}.csv"
        runhistory = load_json(f"{opts.result_dir}/run_{run}/{run}/runhistory.json")
        configs = runhistory["configs"]
        values = runhistory["data"]

        est_graphs = []
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for i in range(1, len(configs) + 1):
            print(f"Config: {i}/{len(configs)}")
            if values[i - 1][4] == np.inf:
                print("Infinity encountered")
                est_graphs.append([])
            else:
                config = Configuration(config_space, configs[str(i)])
                process = ctx.Process(
                    target=wrapper_queue, 
                    args=(datapath, data_type, config, False, run, queue), 
                    daemon=True
                )
                process.start()
                try:
                    graph = queue.get(True, opts.trial_walltime_limit)
                except Empty:
                    process.terminate()
                    graph = []
                est_graphs.append(graph)
                
        save_pickle(est_graphs, f"{opts.result_dir}/run_{run}/{run}/graphs.pkl")

def adjustment_stars_oct(opts):
    """Adjust the sparsity, mb_size, and pool_yhats for terminated configurations.
    
    Parameters
    ----------
    opts: Namespace
        contains commandline arguments
    """
    # result_dir = output/objective/discrete_10_3/method
    method = opts.result_dir.split("/")[-1]
    dataset = opts.result_dir.split("/")[2]
    output_stars = f"output/stars/{dataset}/{method}"
    output_oct = f"output/oct/{dataset}/{method}"
    for run in range(opts.seed, opts.seed + opts.repetitions):
        print(f"StARS run: {run}")
        runhistory = load_json(f"{output_stars}/run_{run}/{run}/runhistory.json")
        configs = runhistory["configs"]
        values = runhistory["data"]

        if os.path.exists(f"{output_stars}/run_{run}/{run}/sparsity.pkl"):
            sparsities = load_pickle(f"{output_stars}/run_{run}/{run}/sparsity.pkl")
            complete_sparsities = [np.nan if val[4] == np.inf else sparsities.pop(0) for val in values]
            save_pickle(complete_sparsities, f"{output_stars}/run_{run}/{run}/sparsity.pkl")
        else:
            sparsities = np.full(len(values), np.nan)
            save_pickle(sparsities, f"{output_stars}/run_{run}/{run}/sparsity.pkl")

        print(f"OCT run: {run}")
        runhistory = load_json(f"{output_oct}/run_{run}/{run}/runhistory.json")
        configs = runhistory["configs"]
        values = runhistory["data"]

        if os.path.exists(f"{output_oct}/run_{run}/{run}/mb_size.pkl"):
            mb_size = load_pickle(f"{output_oct}/run_{run}/{run}/mb_size.pkl")
            pool_yhats = load_pickle(f"{output_oct}/run_{run}/{run}/pool_yhats.pkl")
            complete_mb_size = [np.nan if val[4] == np.inf else mb_size.pop(0) for val in values]
            complete_pool_yhats = [np.nan if val[4] == np.inf else pool_yhats.pop(0) for val in values]
            save_pickle(complete_mb_size, f"{output_oct}/run_{run}/{run}/mb_size.pkl")
            save_pickle(complete_pool_yhats, f"{output_oct}/run_{run}/{run}/pool_yhats.pkl")
        else:
            mb_size = np.full(len(values), np.nan)
            pool_yhats = [[]] * len(values)
            pool_ys = [[]] * int(dataset.split("_")[1])
            save_pickle(mb_size, f"{output_oct}/run_{run}/{run}/mb_size.pkl")
            save_pickle(pool_yhats, f"{output_oct}/run_{run}/{run}/pool_yhats.pkl")
            save_pickle(pool_ys, f"{output_oct}/run_{run}/{run}/pool_ys.pkl")

def run_comparison(opts):
    """Evaluate the configuration with another dataset (test for increasing data samples size).
    
    Parameters
    ----------
    opts: Namespace
        contains commandline arguments
    """
    output_configurators = []
    # data_dir = "data/dataset/discrete_10_3_200"
    dataset = opts.data_dir.split("/")[-1]
    data_type = dataset.split("_")[0]
    # result_dir = "results/oct/continuous_10_3/autocd"
    data = pd.read_csv(f"{opts.result_dir}.csv")
    method = opts.result_dir.split("/")[-1]
    objective = opts.result_dir.split("/")[1]
    config_space = configuration_space(data_type, "autocd", opts.seed)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for run in range(opts.seed, opts.seed + opts.repetitions):
        print(f"Run: {run}")
        datapath = f"{opts.data_dir}/dataset_{run}.csv"
        true_graph = np.array(load_json(f"data/graph/{dataset}/graph_{run}"))

        best_value = data[objective][run]
        best_config = json.loads(data["config"][run])

        if np.isnan(best_value) or not best_config:
            print("Encountered Infinity value")
            output_configurators.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ""))
        else:
            configuration = Configuration(config_space, best_config)
            process = ctx.Process(
                    target=wrapper_queue, 
                    args=(datapath, data_type, configuration, True, run, queue), 
                    daemon=True
                )
            process.start()
            try:
                est_graph = queue.get(True, opts.trial_walltime_limit)
                true_graph = get_cpdag(true_graph)
                shd, accuracy, fp, fn, n_edges = evaluate(true_graph, est_graph)
                output_configurators.append(
                    (best_value, shd, accuracy, fp, fn, n_edges, best_config)
                )
            except Empty:
                process.kill()
                print("Encountered Infinity value")
                output_configurators.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ""))

    dataframe = pd.DataFrame(
        output_configurators,
        columns=[f"{objective}", "shd", "accuracy", "fp", "fn", "edges", "config"]
    )
    os.makedirs(f"results/{objective}/{dataset}", exist_ok=True)
    dataframe.to_csv(f"results/{objective}/{dataset}/{method}.csv", index=False)

    results = bootstrapping(dataframe, opts.sample_size, opts.n_samples)
    os.makedirs(f"bootstrap/{objective}/{dataset}", exist_ok=True)
    results.to_csv(f"bootstrap/{objective}/{dataset}/{method}.csv", index=False)


def main(opts):
    adjustment_stars_oct(parse_args())
    compute_graphs(parse_args())
    run_eval(parse_args())
    run_stars(parse_args())
    run_oct(parse_args())

if __name__ == "__main__":
    main(parse_args())
