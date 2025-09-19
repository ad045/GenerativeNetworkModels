from gnm.fitting.experiment_saving import *
from gnm.fitting.experiment_dataclasses import Experiment
from gnm import defaults, fitting, generative_rules, weight_criteria, evaluation
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

distance_matrix = defaults.get_distance_matrix(device=DEVICE)
binary_consensus_network = defaults.get_binary_network(device=DEVICE)

eta_values = torch.Tensor([1]) #torch.linspace(-5, -1, 1)
gamma_values = torch.Tensor([-1])#torch.linspace(-0.5, 0.5, 1)
num_connections = int( binary_consensus_network.sum().item() / 2 )

binary_sweep_parameters = fitting.BinarySweepParameters(
    eta = eta_values,
    gamma = gamma_values,
    lambdah = torch.Tensor([0.0]),
    distance_relationship_type = ["powerlaw"],
    preferential_relationship_type = ["powerlaw"],
    heterochronicity_relationship_type = ["powerlaw"],
    generative_rule = [generative_rules.MatchingIndex()],
    num_iterations = [num_connections],
)

weighted_sweep_parameters = fitting.WeightedSweepParameters(
    alpha = [0.01],
    optimisation_criterion = [weight_criteria.DistanceWeightedCommunicability(distance_matrix=distance_matrix) ],
)   

num_simulations = 1

sweep_config = fitting.SweepConfig(
    binary_sweep_parameters = binary_sweep_parameters,
    #weighted_sweep_parameters = weighted_sweep_parameters,
    num_simulations = num_simulations,
    distance_matrix = [distance_matrix]    
)

criteria = [ evaluation.ClusteringKS(), evaluation.DegreeKS(), evaluation.EdgeLengthKS(distance_matrix) ]
energy = evaluation.MaxCriteria( criteria )
binary_evaluations = [energy]
#weighted_evaluations = [ evaluation.WeightedNodeStrengthKS(normalise=True), evaluation.WeightedClusteringKS() ]

experiments = fitting.perform_sweep(sweep_config=sweep_config, 
                                binary_evaluations=binary_evaluations, 
                                real_binary_matrices=binary_consensus_network,
                                #weighted_evaluations=weighted_evaluations,
                                save_model = False,
                                save_run_history = False,
                                verbose=True,
                                wandb_logging=True,
                                method='bayesian',
                                num_bayesian_runs=5
)

eval = ExperimentEvaluation()
print(experiments)