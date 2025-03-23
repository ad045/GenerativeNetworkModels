# import torch
# from typing import List, Iterator, Optional, Any, Dict, Union
# from itertools import product
# from jaxtyping import Float, Int, jaxtyped
# from typeguard import typechecked
# from dataclasses import dataclass, field
import pickle
from .experiment_dataclasses import (
    Experiment,
)
import os

class ExperimentEvaluation():
    def __init__(self, path=None):
        if path is None:
            path = './generative_model_experiments'

        self.path = path

    def save_experiment(self, experiment_dataclass:Experiment, experiment_name='gnm_experiment'):

        def strip_string_to_int(string_value):
            final_number = ''
            started = False
            for character in string_value:
                if character in '1234567890':
                    final_number += character
                    started = True
                elif started:
                    break

            return int(final_number)

        # create path to experiment data and index file if it doesn't exist already
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        all_experiments = os.listdir(self.path)
        with_name = [i for i in all_experiments if experiment_name in i]

        if len(with_name) > 0:
            sorted(with_name)
            last_experiment_name = with_name[-1]
            last_experiment_num = last_experiment_name.split('_')[-1]
            last_experiment_num = strip_string_to_int(last_experiment_num)
            experiment_name += f'_{last_experiment_num + 1}'
        else:
            experiment_name += '_1'
        
        # write to pickle file
        experiment_path = os.path.join(self.path, experiment_name + '.pkl')

        with open(experiment_path, 'wb') as pkl_file:
            pickle.dump(experiment_dataclass, pkl_file)

    # view the experiments as a table and save as csv if you want
    def view_experiments():
        pass

    # find experiment given a value of a variable like alpha, 
    # returns the experiment(s) of that value 
    def query_experiments(value=None, by=None):

        # make sure variable exists

        # iterate through index looking for experiments matching criteria

        # return the experiments

        pass
