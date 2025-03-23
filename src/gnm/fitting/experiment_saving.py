# import torch
# from typing import List, Iterator, Optional, Any, Dict, Union
# from itertools import product
# from jaxtyping import Float, Int, jaxtyped
# from typeguard import typechecked
# from dataclasses import dataclass, field
import json
import pickle
from datetime import datetime
from warnings import warn
from dataclasses import fields, asdict
from .experiment_dataclasses import (
    Experiment,
    BinarySweepParameters,
    WeightedSweepParameters
)
import os

class ExperimentEvaluation():
    def __init__(self, path=None, index_file_path=None, variables_to_ignore=[]):
        if path is None:
            path = 'generative_model_experiments'
        
        if index_file_path is None:
            index_file_path = 'gnm_index.json'

        # create path to experiment data and index file if it doesn't exist already
        if not os.path.exists(path):
            os.mkdir(path)

        self.path = path
        self.index_path = os.path.join(self.path, index_file_path)      
        self._refresh_index_file()

        # get the variables we want to save, i.e. alpha, gamma etc (some will be in list format)
        binary_variables_to_save = [f.name for f in fields(BinarySweepParameters)]
        weighted_variables_to_save = [f.name for f in fields(WeightedSweepParameters)]
        variables_to_save = binary_variables_to_save + weighted_variables_to_save
        self.variables_to_save = [i for i in variables_to_save if i not in variables_to_ignore]

    def _refresh_index_file(self):
        if not os.path.exists(self.index_path):
            self._make_index_file()

        with open(self.index_path, "r") as f:
            data = json.load(f)
        
        self.index_file = data

    def _make_index_file(self):
        date = datetime.now()
        date_formatted = date.strftime("%d/%m/%Y")
        all_variables_to_save = self.binary_variables_to_save + self.weighted_variables_to_save
        json_initial_data = {
            'date':date_formatted,
            'experiment_configs':{
                'test_config':{
                    i:'TEST' for i in all_variables_to_save
                }
            }
        }

        with open(self.index_path, "w") as f:
            json.dump(json_initial_data, f, indent=4)

        self._refresh_index_file()

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
        experiment_name = experiment_name + '.pkl'
        experiment_path = os.path.join(self.path, experiment_name)

        with open(experiment_path, 'wb') as pkl_file:
            pickle.dump(experiment_dataclass, pkl_file)

        # may ignore weighted parameters if set to None
        all_config = asdict(experiment_dataclass.run_config.binary_parameters)
        all_config.update(asdict(experiment_dataclass.run_config.weighted_parameters))
        all_config = {key:repr(value) for key, value in all_config.items() if key in self.variables_to_save}

        self.index_file['experiment_configs'][experiment_name] = all_config
        
        # overwrite previous file
        with open(os.path.join(self.path, "gnm_index.json"), "w") as f:
            json.dump(self.index_file, f, indent=4)

        self._refresh_index_file()
        

    # view the experiments as a table and save as csv if you want
    def view_experiments():
        pass

    def _sort_experiments(self, experiments, variable_to_sort_by, get_names_only=False):
        experiment_names = list(experiments.keys())

        # keys = experiment name, values = experiment values of the given variable to sort by
        sorting_dict = {experiment_name:value for experiment_name, value in zip(experiment_names, [experiments[name][variable_to_sort_by] for name in experiment_names])}

        # sort dictionary by values
        sorting_dict = dict(sorted(sorting_dict.items(), key=lambda item: item[1]))

        # create a new dictonary and add experiments based on the order of sorted_experiments
        # but with all the data included in experiments, rather than just the value used for sorting
        sorted_experiments = {}
        for name in list(sorting_dict.keys()):
            sorted_experiments[name] = experiments[name][variable_to_sort_by]

        if get_names_only:
            sorted_experiments = list(sorted_experiments.keys())
        
        return sorted_experiments
    
    def clean_index_file(self):
        pass

    def _ask_loop(self, question):
        answer = None
        question = question + '\ny=confirm, n=exit\n> '
        while answer is None:
            user_input = input(question).lower()
            if user_input == 'y':
                answer = True
            elif user_input == 'n':
                answer = False
            else:
                print('Invalid response. Must be y for yes or n for no.')

        return answer

    def delete_experiment(self, experiment_name, ask_first=True):
        if not experiment_name in self.index_file['experiment_configs']:
            warn(f'Experiment {experiment_name} not found in index file, exiting.')

        if ask_first:
            response = self._ask_loop(f'Are you sure you want to delete experiment {experiment_name}?')
            if response == False:
                print('Aborting....')
                return

        del self.index_file['experiment_configs'][experiment_name]
        tmp_path = os.path.join(self.path, experiment_name)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        print(f'File {tmp_path} deleted and removed from index.')

    def purge_index_file(self):
        pass

    # find experiment given a value of a variable like alpha, 
    # returns the experiment(s) of that value 
    def query_experiments(self, value=None, by=None, limit=float('inf'), verbose=True):

        # get all searchable variables
        all_experiments = self.index_file['experiment_configs']
        if len(all_experiments) == 0:
            warn(f'No experiments saved in index file {self.index_file}')

        first_experiment = list(all_experiments.keys())[0]
        first_experiment_data = all_experiments[first_experiment]
        searchable_variables = list(first_experiment_data.keys())

        # make sure variable provided can be searched
        assert by in searchable_variables, f'Variable {by} not in searchable variables. Must be one of {searchable_variables}'

        # sort by that variable and return list if no value to search for is specified
        if value is None or len(all_experiments) == 1:
            experiments_sorted = self._sort_experiments(experiments=all_experiments, variable_to_sort_by=by, get_names_only=True)
            return_files = self.open_experiments_by_name(experiments_sorted)
            return return_files 
        
        # iterate through index looking for experiments matching criteria
        to_return = []
        experiments_sorted = self._sort_experiments(experiments=all_experiments, variable_to_sort_by=by, get_names_only=False)

        for experiment_name, experiment_value in experiments_sorted.items():
            if experiment_value == value:
                to_return.append(experiment_name)
        
        experiment_data_to_return = self.open_experiments_by_name(to_return)

        if verbose:
            print(f'Found {len(experiment_data_to_return)} item(s) matching: {by} = {value}')
        
        return experiment_data_to_return
        

    def open_experiments_by_name(self, experiment_names):
        if isinstance(experiment_names, str):
            experiment_names = [experiment_names]

        experiments_opened = []
        for name in experiment_names:
            if name == 'test_config':
                continue

            file_path = os.path.join(self.path, name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    experiments_opened.append(f)
            else:
                warn(f'File {name} could not be found in {file_path}. Do you have the right root folder specified?')
        
        if len(experiments_opened) == 1:
            experiments_opened = experiments_opened[0]
        
        return experiments_opened