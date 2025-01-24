from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
import torch
from abc import ABC, abstractmethod

# abstract base class for all other rules
class GenerativeRule(ABC):
    def __init__(self, mode='in'):
        self.mode = mode

        self.modes = {
            'in': lambda matrix: matrix, # no preprocessing needed
            'out': lambda matrix: matrix.T, # transpose and process as 'in'
            'all': self._all # combine transposition and original
        }

        assert self.mode in ['all', 'in', 'out'], f"Mode '{self.mode}' is not supported. Must be one of 'all', 'in', or 'out'."

    # equation components
    @jaxtyped(typechecker=typechecked)
    def _all(self, matrix:Float[torch.Tensor, "num_nodes num_nodes"]):
        # combine incoming and outgoing connections
        matrix = matrix + matrix.T
        matrix = matrix.fill_diagonal_(0)
        return matrix

    # pipeline for applying rule to adjacency matrix
    @jaxtyped(typechecker=typechecked)
    def __call__(self, adjacency_matrix:Float[torch.Tensor, "num_nodes num_nodes"]):
        # get correct rule and exponential/powerlaw functions
        mode_fcn = self.modes[self.mode]

        # clone and remove self-connections
        tmp_mat = adjacency_matrix.clone()
        tmp_mat.fill_diagonal_(0)

        # apply mode (e.g. all, in, out) to account for directional connectivity
        tmp_mat = mode_fcn(tmp_mat)

        # apply rule e.g. matching, clustering etc
        tmp_mat = self.pass_rule(tmp_mat)

        return tmp_mat
    
    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def pass_rule(self, adjacency_matrix:Float[torch.Tensor, "num_nodes num_nodes"]):
        pass


# matching index rule - basic of BCT
class MatchingIndex(GenerativeRule):
    def __init__(self, mode='in', **kwargs):
        super().__init__(mode)

        self.divisors = {
            'mean':self._mean,
            'union':self._union
        }
        self.divisor = kwargs.get('divisor')
        assert self.divisor in self.divisors.keys(), f"Divisor must be one of {self.divisors.keys()}, {self.divisor} not valid"
        
        self.divisor_fcn = self.divisors[self.divisor]

    @jaxtyped(typechecker=typechecked)
    def pass_rule(self, adjacency_matrix:Float[torch.Tensor, "num_nodes num_nodes"]):
        # divisor needed for normalizaion
        devisor_val = self.divisor_fcn(adjacency_matrix)

        # apply normalization, get matching index, remove self-connections
        matching_indices = (adjacency_matrix.T @ adjacency_matrix) / devisor_val
        matching_indices.fill_diagonal_(0)
        return matching_indices
    
    @jaxtyped(typechecker=typechecked)
    def _mean(self, matrix:Float[torch.Tensor, "num_nodes num_nodes"]):
        node_strengths = matrix.sum(dim=0)
        denominator = (node_strengths.unsqueeze(0) + node_strengths.unsqueeze(1) - matrix - matrix.T) / 2
        denominator[denominator == 0] = 1
        return denominator

    @jaxtyped(typechecker=typechecked)
    def _union(self, adjacency_matrix:Float[torch.Tensor, "num_nodes num_nodes"]):
        denominator = torch.max( adjacency_matrix.unsqueeze(1), adjacency_matrix.unsqueeze(2)).sum(dim=0) - adjacency_matrix - adjacency_matrix.T
        # Set the denominator to be 1 whenever it is zero to avoid division by zero
        denominator[denominator == 0] = 1
        return denominator
    

class ClusteringAvg(GenerativeRule):
    def __init__(self, mode='in'):
        super().__init__(mode)

    @jaxtyped(typechecker=typechecked)
    def pass_rule(self, adjacency_matrix:Float[torch.Tensor, "num_nodes num_nodes"]):
        # TODO: Change clustering coef based on directionary and weighted/binary
        # TODO: implement clustering coef myself 
        
        return
        clustering_coef = bct.clustering_coef_bu(adjacency_matrix.cpu().numpy())
        
        pairwise_clustering_coefficient = clustering_coef[:, None] + clustering_coef
        clustering_avg = pairwise_clustering_coefficient / 2
        clustering_avg = torch.Tensor(clustering_avg)
        clustering_avg.fill_diagonal_(0)
        return clustering_avg