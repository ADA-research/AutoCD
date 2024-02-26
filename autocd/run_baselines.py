import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import pprint as pp
import multiprocessing as mp
from queue import Empty
from copy import deepcopy
from datetime import datetime
from utils import load_json
from objective_function.oct import OCT
from objective_function.stars import StARS
from tetrad_utils import configuration_space


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True, help="Directory to the dataset splits"
    )
    parser.add_argument(
        "--objective_function",
        type=str,
        required=True,
        choices=["stars", "oct"],
        help="Objective function: stars or oct",
    )
    parser.add_argument(
        "--walltime_limit",
        type=int,
        default=3600,
        help="Time in seconds that StARS or OCT is allowed to run",
    )
    parser.add_argument(
        "--trial_walltime_limit",
        type=int,
        default=900,
        help="Time in seconds that a trial is allowed to run",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=25,
        help="Number of configurator runs"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()

def run_experiment(objective, config_space, opts):
    """Utilizing random search to identify higher performing configuration.
    
    Parameters
    ----------
    objective: str
        name of the loss function to use
    config_space: ConfigSpace object
        the search space
    opts: Namespace
        contains the commandline arguments
        
    Returns
    -------
    Dictionary containing the best loss, configuration and elapsed time.
    """
    graphs = []
    configs = []
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    remaining_budget = opts.walltime_limit
    trial_budget = opts.trial_walltime_limit
    start = time.time()
    while remaining_budget > 0:
        config = config_space.sample_configuration()
        start_clock = time.time()
        process = ctx.Process(target=objective.graphs, args=(config, queue), daemon=True)
        process.start()
        try:
            graph_set = queue.get(True, trial_budget)
            process.kill() # process.terminate()
        except Empty:
            process.kill() # process.terminate()
            graph_set = None
        graphs.append(graph_set)
        configs.append(config)
        remaining_budget -= time.time() - start_clock

        if remaining_budget < opts.trial_walltime_limit:
            trial_budget = remaining_budget

        flag_oct = opts.objective_function == "oct" and remaining_budget <= opts.trial_walltime_limit // 2
        flag_stars = opts.objective_function == "stars" and remaining_budget <= 60
        if flag_oct or flag_stars:
            break

    print(f"{datetime.now()}: Start {opts.objective_function} baseline")
    value, config = objective.baseline(graphs, configs)

    if not config == {}:
        config = config.get_dictionary()

    output_dict = {
        "Value": value,
        "Configuration": config,
        "Elapsed time": time.time() - start
    }
    return output_dict


if __name__ == "__main__":
    opts = parse_args()
    pp.pprint(vars(opts))

    # data_dir example: "splits/categorical_20_20"
    dataset = opts.data_dir.split("/")[-1]
    data_type = dataset.split("_")[0]
    output_dir = f"output/{opts.objective_function}/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    config_space = configuration_space(data_type, "autocd", opts.seed)

    output_results = []
    for run in range(opts.seed, opts.seed + opts.repetitions):
        print(f"{datetime.now()}: Start run {run}")
        data_info = load_json(f"data/info/{dataset}/info_{run}")
        datapath = f"{opts.data_dir}/seed-{run}/"

        if opts.objective_function == "stars":
            objective = StARS(datapath, data_info, False, run)
        elif opts.objective_function == "oct":
            objective = OCT(datapath, data_info, False, run)
        else:
            raise ValueError(f"This objective function doesn't exist: {opts.objective_function}")

        output_dict = run_experiment(objective, config_space, opts)
        output_results.append(output_dict)

        # Store the results, when it crashes we can start from certain run
        print(f"{datetime.now()}: Store output")
        df = pd.DataFrame(output_results)
        df["Configuration"] = df["Configuration"].apply(json.dumps)
        df.to_csv(f"{output_dir}/baseline_last.csv", index=False)
    
    dataframe = pd.DataFrame(output_results)
    dataframe["Configuration"] = dataframe["Configuration"].apply(json.dumps)
    dataframe.to_csv(f"{output_dir}/baseline_last.csv", index=False)
