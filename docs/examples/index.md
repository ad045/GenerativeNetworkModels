# Example scripts

Here, we provide in detail a number of example scripts which can be used to run GNMs and learn how they work. They all rely on in-package data which can be accessed from gnm.defaults, including binary/weighted connectomes and a distance matrix. 

These examples are for demonstration purposes only and do not go into great depth about the theory behind GNMs. Rather, they outline a number of ways in which models can be generated, fitted, evaluated, and visualised. 

All scripts are written as a Jupyter notebook for ease of use and accessability. 

### [Sweep Example](https://github.com/EdwardJamesYoung/GenerativeNetworkModels/blob/master/docs/examples/sweep_example.ipynb)
This example demonstrates how to perform parameter sweeps using the GNM package. The models are then evaluated aginst a real network. In the context of GNM's, the term 'evaluation' means comparing generated synthetic networks to real networks using metrics like betweeness centrality, clustering coefficients, degree, and edge distribution. It includes:
- Setting up parameter sweeps using `gnm.fitting.SweepConfig` using a linear eta and gamma parameter space.
- Running sweeps with `gnm.fitting.perform_sweep`.
- Finding optimal models across sweeps based on user-set evaluation critereon.

### [Evaluation Example](https://github.com/EdwardJamesYoung/GenerativeNetworkModels/blob/master/docs/examples/graph_model_performance.ipynb)
This script focuses on how to graph the time taken to run models across different devices. Identical models are produced from the BCT library and this toolbox, and are compared based on their time and log time.

### [Experiment Saving Example](https://github.com/EdwardJamesYoung/GenerativeNetworkModels/blob/master/docs/examples/experiment_saving_example.ipynb)
This example demonstrates how to save and query experiments. The experiment saving capacity of the toolbox makes it easy to run many models and track their parameters while easily querying for specific models/results. It includes:
- A basic parameter sweep
- Saving experiments using `ExperimentEvaluation.save_experiments`.
- Querying experiments by specific criteria, such as `generative_rule`.

### [WandB Integration Example](https://github.com/EdwardJamesYoung/GenerativeNetworkModels/blob/master/docs/examples/example_wandb_run.ipynb)
This example shows how to integrate the GNM package with Weights & Biases (WandB) for experiment tracking. It includes:
- Setting up basic parameters for model runs.
- Iterating through parameter combinations for demonstration purposes.
- Using WandB to log and visualize results.

### [Weighted Parameter Sweeps](https://github.com/EdwardJamesYoung/GenerativeNetworkModels/blob/master/docs/examples/weighted_sweep.ipynb)
The weighted sweep example script demonstrates how sweeping through the 'alpha' parameter creates weighted generative models varying in similarity, relative to a real weighted connectome.  

## How to Run the Examples
1. Ensure you have installed the GNM package as described in the [Getting Started Guide](../getting-started.md).
2. Open the Jupyter notebooks or Python scripts in the `docs/examples` directory.
3. Follow the instructions in each example to run the scripts.

These examples are designed to help you understand the functionality of the GNM package and how to apply it to your own datasets.
