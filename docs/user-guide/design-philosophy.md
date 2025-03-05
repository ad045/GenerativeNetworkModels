# Design Philosophy

Here we outline some of the key design choices we made in writing this package, 
to make it easier for users to know what to expect when engaging with the package. 

## Typechecking

*Typechecking* refers automatically verifying that the arguments passed into functions 
look the way that the function expects them to (*checking* that the arguments are of the right *type*).
Throughout the package, we have used extensive (runtime) typechecking wherever possible. 
As a result, it is difficult to use the package incorrectly and still have code run. In other words, 
typechecking acts as a safeguard that prevents arguments from being passed incorrectly. 

Some concrete examples of typechecking we've employed throughout the package:

* **Checking binary adjacency matrices**. The package expects all binary adjacency matrices to be symmetric (so that the corresponding graph is undirect), zero along the diagonal (no self-connections), and contain only ones or zeros (presence or absence of connections). If a function expects a binary adjacency matrix and you accidentally give it a weighted one, it will throw an error rather than failing silently, making it easier to debug code. The function which performs checking of binary adjacency matrices is [`utils.binary_checks`][gnm.utils.binary_checks].
* **Checking weighted adjacency matrices**. Similarly, weighted adjacency matrices are checked to ensure they are symmetric, zero along the diagonal, and non-negative. The function which performs checking of weighted adjacency matrices is [`utils.weighted_checks`][gnm.utils.weighted_checks]. 
* **Tensor shape checking**. The package uses PyTorch as its back-end, meaning that data is handled in the form of *tensors*. We check the shape of tensors using the jaxtyping package. This allows us to check, for example, that distance and adjacency matrices are always square. Again, this prevents incorrect arguments from being passed into functions. 


## Modularity

The package has been designed to be highly modular. Wherever possible, *abstract base classes* are defined with give the interface for specific modular parts of the algorithm. Some specific examples of this are:

* **The base class for generative rules**, [`generative_rules.GenerativeRule`][gnm.generative_rules.GenerativeRule]. This defines the common interface for rules which map from adjacency matrices into affinity factors. 
* **The base class for evaluations**, [`evaluation.EvaluationCriterion`][gnm.evaluation.EvaluationCriterion]. This defines the common interface for methods which compute a measure of fit between a synthetic network and a real one.
* **The base class for weight optimisation criteria**, [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]. This defines the common interface for criteria which are optimised in the fitting of weighted generative models.

This modular structure of the package makes it easy for new users to quickly define and test new generative rules and weight criteria for themselves. See (LINK TO EXAMPLE SCRIPT 1) and (LINK TO EXAMPLE SCRIPT 2) for these, respectively. 

## Dataclasses

Finally, wherever possible, data (parameters, evaluation results, configuration for sweeps) are packaged up into *dataclasses*. This ensures that relevant information is stored together in a single object. Some examples include:

* **Model parameters**. The [`BinaryGenerativeParameters`][gnm.BinaryGenerativeParameters] dataclass stores parameters for binary models, while the [`WeightedGenerativeParameters`][gnm.WeightedGenerativeParameters] dataclass stores parameters for weighted models. 
* **Sweep configurations**. The configuration for a sweep over a parameter space is stored in a [`fitting.SweepConfig`][gnm.fitting.SweepConfig] dataclass. 
* **Experimental results**. Evaluation results are stored in [`fitting.EvaluationResults`][gnm.fitting.EvaluationResults] dataclasses, while the evolution of a generative model over time is stored in a [`fitting.RunHistory`][gnm.fitting.RunHistory] dataclass. [`fitting.Experiment`][gnm.fitting.Experiment] dataclasses package together evaluation results, run histories, the parameter configuration, and the model itself in one place. 

