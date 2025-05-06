# Getting Started

## Installing Package

### Obtaining a local copy of the package
1. Clone the git repository or download from [GitHub](https://github.com/EdwardJamesYoung/GenerativeNetworkModels.git)
    - You can do this either via git:
        - In the command line:
            1. If you haven't already installed git, you can do so via [git-scm.com](https://git-scm.com/)
            2. `cd ./directory/to/install/into`
            3. `git clone https://github.com/EdwardJamesYoung/GenerativeNetworkModels.git`
    - Or download directly from GitHub (Not Recommended):
        1. Download from the same page as a .zip file
        2. Extract the contents of the file into a chosen directory

### Installing the Package
1. If installing into a virtual environment (.venv) or a conda environment (.conda), ensure these are active. Assuming you have already created the environment:
    - Activate a virtual environment using:
        - `.venv/Scripts/activate`
    - Activate a conda environment using:
        - `conda activate name-of-environment`
2. Run `pip install ./directory/to/install/into` in the command line

Done!


## Getting Your Bearings
This is a big package and can be a little intimidating at first. Not to fear! Here's a quick overview of what you can do with the GNM toolbox and how to do it too. 
For a comphrehensive overview of the toolbox, see [PREPRINT CITE HERE]. 

1. Generative Network Models
The 'GNM' in GNM Toolbox stands for Generative Network Model, which is a model of brain development designed to mimic some of the developmental mechanisms going on at a neural level. 
In its simplest form, you can use a binary GNM to grow a network using a cost rule and a wiring rule. These are the basic rules governing network development and are adjusted to create artificial networks that have similar properties to that of an actual brain network. 

![Consensus Network](/docs/images/binary_consensus.png)
<i>Adjacency Matrix of a Binary Consensus Network</i>

The cost rule tells the network what the 'cost' would be of creating a new connection and is based on a distance matrix given during model initialization (see our [sweep example script](https://generative-network-models-toolbox.readthedocs.io/en/latest/examples/sweep_example/)). You can adjust the influence of this parameter using the gamma parameter. 

The wiring rule tells the model how much value it should put into creating a new connection, the direct opposition to the cost rule. The most popular and 'effective' wiring rule is matching index, which basically says that two neurons are more likely to be connected if they share many of the same neighbours. The extent of the rule can be moderated using the eta parameter, and you can implement a number of other rules using this package which you can find [here](https://generative-network-models-toolbox.readthedocs.io/en/latest/api-reference/generative-rules/).

The economic trade-off between these rules means that we can explore how different parameters and rules act in unison to create a network, with the goal often being creating something that looks as close to the brain as possible. There are many other varieties beyond binary models, like weighted and heterochronicity-based models, which you can have a look at in our [example scripts page](https://generative-network-models-toolbox.readthedocs.io/en/latest/examples/) or within the API reference guide. 

2. Model Fitting
So we're able to create networks that have many of the properties of the brain. This is amazing, but what if we want to find properties of a <i>specific</i> brain? This is where model sweeps come in. We've mentioned <i>eta</i> $\eta$ and <i>gamma</i> $\gamma$ so far, and that you can play with these parameters to change their influence in the model. By running through a sequence of each and comparing the final model with that of a real brain network in the form of an adjacency matrix, you can work out what parameters of the model will create properties best representative of reality. Again, this is where our example scripts come in really helpful if you want to learn more. 
