from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
import torch
import bct

class GenerativeRule():
    def __init__(self, rule, mode='in', divisor='mean'):
        self.mode = mode
        self.divisor = divisor
        self.rule = rule

        self.rules = {
            'matching_index': self._matching_index,
            'clu_avg': self._clustering_avg
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
        assert self.divisor in self.devisors.keys(), f"Divisor '{self.divisor}' is not supported. Must be one of 'mean' or 'union'."
        assert self.rule in self.rules.keys(), f"Rule '{self.rule}' is not supported. Must be one of {self.rules.keys()}."
    
    # set of rules 
    def _clustering_avg(self, adjacency_matrix:torch.Tensor):
        # TODO: Change clustering coef based on directionary and weighted/binary
        clustering_coef = bct.clustering_coef_bu(adjacency_matrix.cpu().numpy())
        pairwise_clustering_coefficient = clustering_coef[:, None] + clustering_coef
        clustering_avg = pairwise_clustering_coefficient / 2
        clustering_avg = torch.Tensor(clustering_avg)
        clustering_avg.fill_diagonal_(0)
        return clustering_avg

    def _matching_index(self, adjacency_matrix:torch.Tensor):
        # divisor needed for normalizaion
        devisor_fcn = self.devisors[self.divisor]
        devisor_val = devisor_fcn(adjacency_matrix)

        # apply normalization, get matching index, remove self-connections
        matching_indices = (adjacency_matrix.T @ adjacency_matrix) / devisor_val
        matching_indices.fill_diagonal_(0)
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
        denominator = torch.max( matrix.unsqueeze(1), matrix.unsqueeze(2)).sum(dim=0) - matrix - matrix.T
        # Set the denominator to be 1 whenever it is zero to avoid division by zero
        denominator[denominator == 0] = 1
        return denominator

    # pipeline for applying rule to adjacency matrix
    def __call__(self, adjacency_matrix:torch.Tensor):
        # get correct rule and exponential/powerlaw functions
        mode_fcn = self.modes[self.mode]
        rule_fcn = self.rules[self.rule]

        # clone and remove self-connections
        tmp_mat = adjacency_matrix.clone()
        tmp_mat.fill_diagonal_(0)

        # apply mode (e.g. all, in, out) to account for directional connectivity
        tmp_mat = mode_fcn(tmp_mat)

        # apply rule e.g. matching, clustering etc
        tmp_mat = rule_fcn(tmp_mat)

        return tmp_mat
    
