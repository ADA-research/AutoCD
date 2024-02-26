import os
import time
import argparse
import numpy as np
import pprint as pp
from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPO
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
        "--algorithm", 
        type=str,
        default="autocd",
        choices=["autocd", "pc", "fges", "lingam", "golem"],
        help="Search space of the method/algorithm"
    )
    parser.add_argument(
        "--objective_function",
        type=str,
        required=True,
        choices=["stars", "oct"],
        help="Objective function of SMAC",
    )
    parser.add_argument(
        "--walltime_limit",
        type=int,
        default=3600,
        help="Time in seconds that SMAC is allowed to run",
    )
    parser.add_argument(
        "--trial_walltime_limit",
        type=int,
        default=900,
        help="Time in seconds that a trial is allowed to run",
    )
    parser.add_argument(
        "-deterministic",
        action="store_true",
        help="Set true to only use one random seed",
    )
    parser.add_argument(
        "-start_pc",
        action="store_true",
        help="Set true to only use one random seed",
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

def main(opts):
    """Run SMAC to identify higher performing configuration.
    
    Parameters
    ----------
    opts: Namespace
        containing the commandline arguments
    """
    pp.pprint(vars(opts))

    # data_dir example: "splits/continuous_5_2"
    dataset = opts.data_dir.split("/")[-1]
    data_type = dataset.split("_")[0]
    autocd = True if opts.algorithm == "autocd" else False
    # output_dir = f"output/{opts.objective_function}/{dataset}/{opts.algorithm}"
    output_dir=f"output/{opts.objective_function}/{dataset}/{opts.algorithm}_pc"
    config_space = configuration_space(data_type, opts.algorithm, opts.seed)

    for run in range(opts.seed, opts.seed + opts.repetitions):
        if dataset == "mixed_9_shen":
            data_info = load_json(f"data/info/{dataset}")
        else:
            data_info = load_json(f"data/info/{dataset}/info_{run}")
        datapath = f"{opts.data_dir}/seed-{run}/"
        scenario = Scenario(
            configspace=config_space,
            name=f"run_{run}",
            output_directory=output_dir,
            deterministic=opts.deterministic,
            walltime_limit=opts.walltime_limit,
            trial_walltime_limit=opts.trial_walltime_limit,
            use_default_config=opts.start_pc,
            seed=run
        )

        initial_design = HPO.get_initial_design(
            scenario=scenario, n_configs=5
        )
        
        if opts.objective_function == "stars":
            objective = StARS(datapath, data_info, autocd, run)
        elif opts.objective_function == "oct":
            objective = OCT(datapath, data_info, autocd, run)
        else:
            raise ValueError(f"This objective function doesn't exist: {opts.objective_function}")

        smac = HPO(
            scenario, objective.tuning, initial_design=initial_design, overwrite=False
        )

        smac.optimize()
        # Results are stored in /output/objective_function/dataset/algorithm/run_x/x/

if __name__ == "__main__":
    main(parse_args())
