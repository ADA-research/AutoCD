from copy import deepcopy
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from search_space import *


ALGORITHMS = {
    "pc": PC,
    "pcstable": PCStable,
    "cpc": CPC,
    "cpcstable": CPCStable,
    "fges": FGES,
    "icalingam": ICALiNGAM,
    "boss": BOSS,
    "grasp": GRaSP,
    "dagma": DAGMA,
    "directlingam": DirectLiNGAM,
    "golem": GOLEM
}

def wrapper_cd(datapath, data_type, config, cpdag, seed):
    """Build and estimate the causal model and causal adjacency matrix using
    the given data and configuration.

    Parameters
    ----------
    data: str
        datapath to the dataset
    data_type: str
        either categorical, continuous or mixed
    config: Configuration object
        contains a configuration of hyperparameters
    seed: int   
        random seed

    Returns
    -------
    Estimated pdag adjacency matrix.
    """
    params = {}
    config_copy = deepcopy(config)
    param_dict = config_copy.get_dictionary()
    algorithm = param_dict["algorithm"]
    del param_dict["algorithm"]

    for param, value in param_dict.items():
        param = param.replace(f"{algorithm}_", "")
        params[param] = value

    model = ALGORITHMS[algorithm](data_type, seed)
    G_dag, G_cpdag = model.estimate(datapath, params)
    if cpdag:
        return G_cpdag
    else:
        return G_dag

def configuration_space(data_type, method, seed):
    """Create configuration space for given method (i.e. AutoCD).

    Parameters
    ----------
    data_type: str
        the data type of the dataset
    method: str
        the method to use either pc, fges, lingam, golem or autocd
    seed: int
        random seed

    Returns
    -------
    The configuration space for given method.
    """
    if method == "pc":
        algorithm_choice = CategoricalHyperparameter(
            "algorithm", choices=["pc"]
        )
        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(algorithm_choice)
        config_space.add_configuration_space(
            "pc",
            PC(data_type, seed).get_cs(),
            delimiter="_",
            parent_hyperparameter={"parent": algorithm_choice, "value": "pc"},
        )
    elif method == "fges":
        algorithm_choice = CategoricalHyperparameter(
            "algorithm", choices=["fges"]
        )
        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(algorithm_choice)
        config_space.add_configuration_space(
            "fges",
            FGES(data_type, seed).get_cs(),
            delimiter="_",
            parent_hyperparameter={"parent": algorithm_choice, "value": "fges"},
        )
    elif method == "lingam":
        if not data_type == "continuous":
            raise ValueError(f"This data type doesn't work with ICA-LiNGAM: {data_type}")
        else:
            algorithm_choice = CategoricalHyperparameter(
                "algorithm", choices=["icalingam"]
            )
            config_space = ConfigurationSpace()
            config_space.add_hyperparameter(algorithm_choice)
            config_space.add_configuration_space(
                "icalingam",
                ICALiNGAM(data_type, seed).get_cs(),
                delimiter="_",
                parent_hyperparameter={"parent": algorithm_choice, "value": "icalingam"},
            )
    elif method == "golem":
        if not data_type == "continuous":
            raise ValueError(f"This data type doesn't work with GOLEM: {data_type}")
        else:
            algorithm_choice = CategoricalHyperparameter(
                "algorithm", choices=["golem"]
            )
            config_space = ConfigurationSpace()
            config_space.add_hyperparameter(algorithm_choice)
            config_space.add_configuration_space(
                "golem",
                GOLEM(data_type, seed).get_cs(),
                delimiter="_",
                parent_hyperparameter={"parent": algorithm_choice, "value": "golem"},
            )
    else: # using autocd
        if data_type == "continuous":
            algorithm_choice = CategoricalHyperparameter(
                "algorithm", choices=list(ALGORITHMS.keys())
            )
            config_space = ConfigurationSpace()
            config_space.add_hyperparameter(algorithm_choice)
            for name, algorithm in ALGORITHMS.items():
                config_space.add_configuration_space(
                    name,
                    algorithm(data_type, seed).get_cs(),
                    delimiter="_",
                    parent_hyperparameter={"parent": algorithm_choice, "value": name},
                )
        else:
            choices = deepcopy(ALGORITHMS)
            choices.pop("dagma")
            choices.pop("directlingam")
            choices.pop("icalingam")
            choices.pop("golem")
            algorithm_choice = CategoricalHyperparameter(
                "algorithm", choices=list(choices.keys())
            )
            config_space = ConfigurationSpace()
            config_space.add_hyperparameter(algorithm_choice)
            for name, algorithm in choices.items():
                config_space.add_configuration_space(
                    name,
                    algorithm(data_type, seed).get_cs(),
                    delimiter="_",
                    parent_hyperparameter={"parent": algorithm_choice, "value": name},
                )

    return config_space
