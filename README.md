# Automated Causal Discovery

AutoML system that does algorithm selection and hyperparameter optimization for causal discovery algorithms. It uses the Bayesian optimization framework SMAC (Sequential Model-based Algorithm Configuration) to efficiently find a good performing configuration which is sampled from the search space. The search space contains causal discovery algorithms and their respective hyperparameters from the packages: Tetrad and gCastle.

## Dependencies

This package uses Python and Java JAR files.

### Install

To obtain the Java JAR files, py-tetrad and causal-cmd need to be installed. This can be done by following the installation instruction [cmu-phil/py-tetrad](https://github.com/cmu-phil/py-tetrad/) [bd2kccd/causal-cmd](https://github.com/bd2kccd/causal-cmd/) or a summarized step-by-step installation guide below:

* Install JDK version 17+ and set JAVA_HOME to path of Java installation. For help, see [Set up Java for Tetrad](https://github.com/cmu-phil/tetrad/wiki/Setting-up-Java-for-Tetrad)
* Install JPype package 
```bash
pip install Jpype1
```
* Clone the github repository py-tetrad and place the JAR file in the directory `py-tetrad/pytetrad` under the path `repo/autocd/pytetrad` and make sure that the JAR file is named `tetrad-current.jar`
```bash
git clone https://github.com/cmu-phil/py-tetrad/
```
* Download the JAR file from github repository causal-cmd and store JAR file under `repo/autocd/causalcmd` and make sure that the JAR file is named `causal-cmd-1.11.0.jar` [https://github.com/bd2kccd/causal-cmd/tree/release-v1.11.0](https://github.com/bd2kccd/causal-cmd/tree/release-v1.11.0)
```bash
mkdir causalcmd
```

Instruction to install SMAC can be found in [SMAC3](https://github.com/automl/SMAC3). This repository is only tested with Python 3.9. The file `python_requirements.txt` contains other Python packages (e.g. numpy and pandas) and can be installed:
```bash
cd build-utils
pip install -r python_requirements.txt
```

## Usage

### Generating data

The data, ground truth graph, data information, and data splits can be generated by the following command:
```bash
python generate_data.py --nodes 10 --degree 3 --instances 1000 --data_type mixed --repetition 25 --subsamples 20 --folds 10 --seed 0
```
This command creates a graphical model with 10 nodes, average node degree 3, and mixed variables. This model is used to simulate a dataset with 1000 instances using seed 0. This is repeated 25 times such that we have 25 different datasets. The datasets are then split using subsampling (20 subsamples) and KFold cross-validation (10 training-validation sets). These splits are stored in "splits/mixed_10_3".

### Running AutoCD

The following command can be used to run AutoCD:
```bash
python run.py --data_dir splits/mixed_10_3 --algorithm autocd --objective_function oct --walltime_limit 3600 --trial_walltime_limit 900 -deterministic --repetitions 25 --seed 0
```
This command runs AutoCD with splits located in "splits/mixed_10_3" using OCT as loss function with a budget of 1 hour, each trial terminates after 15 min and only one seed is used. After 25 hours (25 repetitions), the results are stored in the folder "output/oct/mixed_10_3/autocd" which contains 25 folders for each run.

### Evaluation

The output results of AutoCD are evaluated using the following command:
```bash
python eval.py --data_dir data/dataset/mixed_10_3 --result_dir output/oct/mixed_10_3/autocd --sample_size 5 --n_samples 1000 --trial_walltime_limit 900 --repetitions 25 --seed 0 
```
This command makes adjustments to the sparsity.pkl, mb_size.pkl, pool_yhats.pkl files in the result directory that will be used for AutoCD+ (AutoCD with a post-hoc correction). It computes an additional file graphs.pkl in the result directory to make evaluation faster. It then runs the evaluation using the best found configuration of AutoCD and it runs the evaluation using the best found configuration of AutoCD+. The results are stored in the folder "results/oct/mixed_10_3" and the bootstrap results are stored in the folder "bootstrap/oct/mixed_10_3".

### Visualizations

The Jupyter notebook under visualizations can be used to visualize the AutoCD results.
