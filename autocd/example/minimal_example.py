"""
Minimal example of applying AutoCD on the mixed dataset with 5 nodes and node degree 2.
"""

import os
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
from queue import Empty
from smac import Scenario
from ConfigSpace import Configuration
from smac import HyperparameterOptimizationFacade as HPO
from utils import load_json
from objective_function.oct import OCT
from objective_function.stars import StARS
from tetrad_utils import wrapper_cd, configuration_space

class MinimalExample:
    def __init__(self, data_dir, algorithm, obj, walltime_limit, trial_limit, deterministic, start_pc, seed):
        self.data_dir = data_dir
        self.algorithm = algorithm
        self.objective_function = obj
        self.walltime_limit = walltime_limit
        self.trial_walltime_limit = trial_limit
        self.deterministic = deterministic
        self.start_pc = start_pc
        self.seed = seed

    def run(self):
        """Run SMAC."""
        dataset = self.data_dir.split("/")[-1]
        data_type = dataset.split("_")[0]
        autocd = True if self.algorithm == "autocd" else False
        config_space = configuration_space(data_type, self.algorithm, self.seed)

        algorithm = "autocd_pc" if self.algorithm == "autocd" and self.start_pc else self.algorithm
        output_dir=f"test/{self.objective_function}/{dataset}/{algorithm}"

        data_info = load_json(f"data/info/{dataset}/info_{self.seed}")
        datapath = f"{self.data_dir}/seed-{self.seed}/"
        scenario = Scenario(
            configspace=config_space,
            name=f"run_{self.seed}",
            output_directory=output_dir,
            deterministic=self.deterministic,
            walltime_limit=self.walltime_limit,
            trial_walltime_limit=self.trial_walltime_limit,
            use_default_config=self.start_pc,
            seed=self.seed
        )

        initial_design = HPO.get_initial_design(
            scenario=scenario, n_configs=5
        )

        if self.objective_function == "stars":
            objective = StARS(datapath, data_info, autocd, self.seed)
        elif self.objective_function == "oct":
            objective = OCT(datapath, data_info, autocd, self.seed)
        else:
            raise ValueError(f"This objective function doesn't exist: {self.objective_function}")

        smac = HPO(
            scenario, objective.tuning, initial_design=initial_design, overwrite=False
        )

        smac.optimize()
