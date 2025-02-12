# Package Overview

The main models implemented within the package are found in the GenerativeNetworkModel class. 

Additionally, there are four main sub-packages. 
Below, we give the main functionality of each subpackage. 
See the full docs pages for the functions and classes available in each sub-package.  
1. [gnm.generative_rules](#gnmgenerative_rules)
2. [gnm.weight_criteria](#gnmweight_criteria)
3. [gnm.fitting](#gnmfitting)
4. [gnm.defaults](#gnmdefaults)
5. [gnm.utils](#gnmutils)

## [model](model.md)

There are two varieties of models implemented within the package:

1. **Binary models** - these capture only the presence and absence of connections within a network, without capturing their strength
2. **Weighted models** - these additionally capture the strengths of the connections within the network. 

Both of these are implemented within the GenerativeNetworkModel class. 

## [gnm.generative_rules](generative-rules.md)

The gnm.generative_rules sub-package contains a collection of different generative rules that can be used to grow 
and develop the generative network. 

## [gnm.weight_criteria](weight-criteria.md)

The gnm.weight_criteria sub-package contains a collection of optimisation criteria that can be used to update the
weights of a weighted generative network model. 

## [gnm.fitting](fitting.md)

The gnm.fitting sub-package contains classes and functions relating the fitting and evaluating 
generative models and parameters.

## [gnm.defaults](defaults.md)

The gnm.defaults sub-package contains default values that can be used to run experiments out-of-the-box

## [gnm.utils](utils.md)

Finally, the gnm.utils sub-package contains other useful functionality for working with the toolbox. 