import os
import subprocess
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from search_space.utils import graph_to_matrix


class DirectLiNGAM:
    """Causal discovery with direct linear non-Gaussian acyclic model.

    References
    ----------
    Shimizu, Shohei and Inazumi, Takanori and Sogawa, Yasuhiro and Hyvarinen, Aapo and 
    Kawahara, Yoshinobu and Washio, Takashi and Hoyer, Patrik O and Bollen, Kenneth and 
    Hoyer, Patrik,
    "DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model", 
    JMLR 2011.

    Hyvärinen, Aapo and Smith, Stephen M,
    "Pairwise likelihood ratios for estimation of non-Gaussian structural equation models", 
    JMLR 2013.

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
        try:
            pd.read_csv(datapath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file {datapath} was not found. Please make sure the file exists.")
            
        os.makedirs("causalcmd/", exist_ok=True)
        prefix = "causalcmd/direct_lingam_" + str(datetime.now()).replace(" ", "_")
        if config["score"] == "sem-bic-score":
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "direct-lingam",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--score", config["score"],
                "--penaltyDiscount", str(config["penalty"]),
                "--semBicStructurePrior", str(config["structure"]),
                "--seed", str(self.seed),
                "--skip-validation",
                "--make-bidirected-undirected",
                "--prefix", prefix
            ])
        else:
            subprocess.run([
                "java", "-Xmx10g", "-jar", "causalcmd/causal-cmd-1.11.0.jar",
                "--algorithm", "direct-lingam",
                "--data-type", self.data_type,
                "--dataset", datapath,
                "--delimiter", "comma",
                "--score", config["score"],
                "--penaltyDiscount", str(config["penalty"]),
                "--structurePrior", str(config["structure"]),
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
        if self.data_type == "continuous":
            score = CategoricalHyperparameter(
                name="score",
                choices=["sem-bic-score", "cg-bic-score", "dg-bic-score"],
                default_value="sem-bic-score"
            )
            penalty = UniformFloatHyperparameter(
                name="penalty", lower=0.0, upper=2.0, default_value=1.0
            )
            structure = UniformFloatHyperparameter(
                name="structure", lower=0.0, upper=2.0, default_value=0.0
            )
            config_space.add_hyperparameters(
                [score, penalty, structure]
            )
            return config_space
