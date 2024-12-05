from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
import torch

class GenerativeRule():
    def __init__(self, rule, mode='in', divisor='mean'):
        self.mode = mode
        self.divisor = divisor
        self.rule = rule

        self.rules = {
            'matching_index': self.matching_index
        }

        self.devisors = {
            'mean':self._mean,
            'union':self._union
        }

        self.modes = {
            'in': lambda matrix: matrix, # no preprocessing needed
            'out': lambda matrix: matrix.T, # transpose and process as 'in'
            'all': self._all # combine transposition and original
        }

        assert self.mode in ['all', 'in', 'out'], f"Mode '{self.mode}' is not supported. Must be one of 'all', 'in', or 'out'."
        assert self.divisor in self.divisor_fcs.keys(), f"Divisor '{self.divisor}' is not supported. Must be one of 'mean' or 'union'."
        assert self.rule in self.rules.keys(), f"Rule '{self.rule}' is not supported. Must be one of {self.rules.keys()}."


    # set of rules 

    @jaxtyped(typechecker=typechecked)
    def matching_index(self, adjacency_matrix) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        num_nodes = adjacency_matrix.shape[0]
        matching_indices = torch.zeros((num_nodes, num_nodes), dtype=adjacency_matrix.dtype)
        return matching_indices
    

    # equation components

    def _all(self, matrix:torch.Tensor):
        # combine incoming and outgoing connections
        matrix = matrix + matrix.T
        matrix = matrix.fill_diagonal_(0)
        return matrix

    def _mean(self, matrix:torch.Tensor):
        node_strengths = matrix.sum(dim=0)
        denominator = (node_strengths.unsqueeze(0) + node_strengths.unsqueeze(1) - matrix - matrix.T) / 2
        denominator[denominator == 0] = 1
        return denominator

    def _union(self, matrix):
        denominator = torch.max( matrix.unsqueeze(1), matrix.unsqueeze(2) ).sum(dim=0) - matrix - matrix.T
        # Set the denominator to be 1 whenever it is zero to avoid division by zero
        denominator[denominator == 0] = 1
        return denominator
    
    def _matching_rule(self, combined_adjacency_matrix, denominator):
        matching_indices = (combined_adjacency_matrix.T @ combined_adjacency_matrix) / denominator
        matching_indices.fill_diagonal_(0)
        return matching_indices


    # pipeline for applying rule to adjacency matrix
    def __call__(self, adjacency_matrix:torch.Tensor):
        # clone and create diagnonal
        tmp_mat = adjacency_matrix.clone()
        tmp_mat = tmp_mat.fill_diagonal(0)

        # apply mode (e.g. all, in, out) to account for directional connectivity
        mode_fcn = self.modes[self.mode]
        tmp_mat = mode_fcn(tmp_mat)

        # get devisor for normalization
        devisor_fcn = self.devisors[self.divisor]
        devisor_val = devisor_fcn(tmp_mat)
        
        # apply rule (e.g. matching etc)
        rule_fcn = self.rules(self.rule)
        tmp_mat = rule_fcn(tmp_mat, devisor_val)

        return tmp_mat
    
