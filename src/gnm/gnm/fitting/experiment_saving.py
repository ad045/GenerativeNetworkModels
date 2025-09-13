import json
import pickle
from datetime import datetime
from warnings import warn
import uuid
from dataclasses import fields, asdict
from .experiment_dataclasses import (
    Experiment,
    BinarySweepParameters,
    WeightedSweepParameters
)
import os
from pathlib import Path
import torch

"""
This module provides functionality for managing and saving experiment data for generative network models. 
It includes the `ExperimentEvaluation` class, which handles saving experiments, managing an index file, 
querying experiments, and other related operations.

Classes:
    - ExperimentEvaluation: Handles saving, querying, and managing experiment data and configurations.

Usage:
    The `ExperimentEvaluation` class can be used to:
    - Save experiments and their configurations.
    - Query experiments based on specific variables and values.
    - Manage an index file that tracks experiment configurations.
    - Delete or clean up experiment data.

Example:
    evaluator = ExperimentEvaluation(path="experiment_data", index_file_path="index.json")
    evaluator.save_experiments([experiment1, experiment2])
    results = evaluator.query_experiments(value=0.5, by="alpha")
"""

class ExperimentEvaluation():
    """
    The `ExperimentEvaluation` class provides functionality for managing and saving experiment data for generative network models. 
    It handles creating directories, managing index files, saving experiment configurations, and querying experiments.
    
    Attributes:
        - path (str): Directory where experiment data is stored. Defaults to 'generative_model_experiments'.
        - index_path (str): Path to the index file that tracks experiment configurations.
        - variables_to_save (list): List of variables to save, excluding those specified in `variables_to_ignore`.
    
    Methods:
        - __init__(path=None, index_file_path=None, variables_to_ignore=[]): Initializes the class, sets up paths, and prepares the index file.
        - _refresh_index_file(): Reloads the index file from disk or creates it if it doesn't exist.
        - _make_index_file(): Creates a new index file with initial data.
        - save_experiments(experiments): Saves a list of `Experiment` objects to disk.
        - _save_experiment(experiment_dataclass, experiment_name='gnm_experiment'): Saves a single experiment and updates the index file.
        - view_experiments(): Placeholder for viewing experiments as a table or saving them as CSV.
        - _sort_experiments(experiments, variable_to_sort_by, get_names_only=False): Sorts experiments by a specified variable.
        - clean_index_file(): Placeholder for cleaning up the index file.
        - _ask_loop(question): Prompts the user for confirmation with a yes/no question.
        - delete_experiment(experiment_name, ask_first=True): Deletes an experiment and removes it from the index file.
        - purge_index_file(): Placeholder for purging the index file.
        - _is_similar_wording(variable_word, verbose=True): Suggests the most similar variable name if a given name is not found.
        - query_experiments(value=None, by=None, limit=float('inf'), verbose=True): Queries experiments based on a variable and value.
        - open_experiments_by_name(experiment_names): Opens experiments by their names and returns their data.
    
    Usage:
        evaluator = ExperimentEvaluation(path="experiment_data", index_file_path="index.json")
        evaluator.save_experiments([experiment1, experiment2])
        results = evaluator.query_experiments(value=0.5, by="alpha")
    """

    def __init__(self, path=None, index_file_path=None, variables_to_ignore=[], save=True):
        self.save = save

        if path is None:
            path = "generative_model_experiments" 
            # path = "/Users/adrian/Documents/01_projects/14_4D_lab/output/02_gnm_est_new" # TODO: remove this hardcoded path
            warn('No path specified, using default "generative_model_experiments"')
   
        if index_file_path is None:
            index_file_path = 'gnm_index.json'

        # create path to experiment data and index file if it doesn't exist already
        # if not os.path.exists(path) and save:
        #     os.mkdir(path)
        if self.save:
            os.makedirs(self.path, exist_ok=True)  # <- instead of os.mkdir
            self._refresh_index_file()

        self.path = path
        self.index_path = os.path.join(self.path, index_file_path)      

        # get the variables we want to save, i.e. alpha, gamma etc (some will be in list format)
        binary_variables_to_save = [f.name for f in fields(BinarySweepParameters)]
        weighted_variables_to_save = [f.name for f in fields(WeightedSweepParameters)]
        variables_to_save = binary_variables_to_save + weighted_variables_to_save
        self.variables_to_save = [i for i in variables_to_save if i not in variables_to_ignore]

        if self.save:
            self._refresh_index_file()

    def _refresh_index_file(self):
        if not os.path.exists(self.index_path):
            self._make_index_file()

        with open(self.index_path, "r") as f:
            data = json.load(f)
        
        self.index_file = data

    def _make_index_file(self):
        
        os.makedirs(self.path, exist_ok=True)  # Ensure the directory exists
        
        date = datetime.now()
        date_formatted = date.strftime("%d/%m/%Y")
        json_initial_data = {
            'date':date_formatted,
            'experiment_configs':{
                'test_config':{
                    i:'TEST' for i in self.variables_to_save
                }
            }
        }

        with open(self.index_path, "w") as f:
            json.dump(json_initial_data, f, indent=4)

        self._refresh_index_file()

    """
    Saves a list of experiments to disk. Each experiment is an instance of the Experiment dataclass.
    The experiments are saved in a specified directory, and the index file is updated accordingly.
    """
    def save_experiments(self, experiments:list[Experiment]):
        if not isinstance(experiments, list):
            experiments = [experiments]
        for experiment in experiments:
            self._save_experiment(experiment)

    def _save_experiment(self, experiment_dataclass:Experiment, experiment_name='gnm_experiment'):

        # Double-check the folder exists 
        if self.save: 
            os.makedirs(self.path, exist_ok=True)  
            
        # TODO: This is a rough fix. It solved for me an error though, that happened when running the script for the first time (w.o. gnm_experiment existing yet)
        os.makedirs(self.path, exist_ok=True)  # Ensure the directory exists 
            
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
        
        other_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/src/connectome_analysis/wandb") # TODO: REMOVE THIS HARD CODED PATH
        # all_experiments = os.listdir(self.path)#
        all_experiments = os.listdir(other_path)
        with_name = [i for i in all_experiments if experiment_name in i]

        if len(with_name) > 0:
            sorted(with_name)
            last_experiment_name = with_name[-1]
            last_experiment_num = last_experiment_name.split('_')[-1]
            last_experiment_num = strip_string_to_int(last_experiment_num)
            experiment_name += f'_{uuid.uuid4().hex}'
        else:
            experiment_name += '_1'
        
        if self.save:
            # write to pickle file
            experiment_name = experiment_name + '.pkl'
            experiment_path = os.path.join(self.path, experiment_name)
            experiment_path = os.path.join("/Users/adrian/Documents/01_projects/14_4D_lab/output/02_gnm_estimation", experiment_name, "_todo.pkl") # TODO: remove this hardcoded path

            with open(experiment_path, 'wb') as pkl_file:
                pickle.dump(experiment_dataclass, pkl_file)

        # may ignore weighted parameters if set to None
        all_config = asdict(experiment_dataclass.run_config.binary_parameters)
        if experiment_dataclass.run_config.weighted_parameters is not None:
            all_config.update(asdict(experiment_dataclass.run_config.weighted_parameters))
    
        # de-tensor floating and int values
        formatted_config = {}
        for key, value in all_config.items():
            if isinstance(value, torch.Tensor):
                formatted_config[key] = value.item()
            elif not isinstance(value, (str, int, float, list)) and hasattr(value, "__class__"):
                try:
                    class_name = value.__class__.__name__
                    formatted_config[key] = class_name
                except:
                    warn(f'Attribute {value} could not be saved - no name or class instance found.')
            elif isinstance(value, list):
                formatted_config[key] = repr(value)
            elif isinstance(value, (int, float, str)):
                formatted_config[key] = value

        # return names and values of parameters if save is False - mainly used for wandb
        if not self.save:
            return formatted_config

        self.index_file['experiment_configs'][experiment_name] = formatted_config
        
        # overwrite previous file
        with open(self.index_path, "w") as f: # replaced "os.path.join(self.path," with "self.index_path" -> consistency 
            json.dump(self.index_file, f, indent=4)
            
        
        # #### NEW 
        
        # # Create experiment dir before serializing per run 
        # if experiment_name:
        #     exp_dir = self.config.paths.gnm_output_dir / experiment_name
        # else:
        #     import time 
        #     timestamp = time.strftime("%Y%m%d_%H%M%S")
        #     exp_dir = self.config.paths.gnm_output_dir / f"gnm_sweep_{timestamp}"
        # exp_dir.mkdir(parents=True, exist_ok=True)

        # # Save all runs to a JSON file
        # all_runs_path = exp_dir / "all_runs.jsonl"

        # # Try to import wandb; if unavailable or not initialized, skip logging 
        # try:
        #     import wandb
        #     _wandb_ok = True
        # except Exception:
        #     _wandb_ok = False
        
        # # Extract params and energy 
        # for i, exp in enumerate(experiments):
        #     try:
        #         # paramters
        #         bp = exp.run_config.binary_parameters  # BinaryGenerativeParameters
        #         eta   = float(bp.eta)
        #         gamma = float(bp.gamma)
        #         rule  = str(bp.generative_rule)
                
        #         # energy 
        #         energies = {}
        #         try:
        #             energy = float(exp.evaluation_dict[evaluation_criteria])
        #             for i in range(len(exp.evaluation_results.binary_evaluations)):
        #                 energies[i] = exp.evaluation_results.binary_evaluations[i]
        #             # energies = exp.evaluation_results.binary_evaluations["MaxCriteria(DegreeKS, ClusteringKS)"] # change this
        #         except Exception:
        #             # fallback if the keying differs in your GNM version: remove? 
        #             energy = float(exp.evaluation_dict.get(str(evaluation_criteria), float("nan")))

        #         run_record = {
        #             "idx": i,
        #             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        #             "method": "bayesian",                      # matches call above
        #             "eta": eta,
        #             "gamma": gamma,
        #             "generative_rule": rule,
        #             "num_iterations": int(target_network.sum().item() // 2),
        #             "num_simulations": self.config.gnm.num_simulations,
        #             "energy": energy,
        #             "individual_energies": energies, 
        #             "device": self.config.gnm.device,
        #         }

        #         # Append JSON line (one run per line)
        #         with open(all_runs_path, "a") as f:
        #             f.write(json.dumps(run_record) + "\n")

        #         # Explicit W&B logging
        #         if _wandb_ok:
        #             try:
        #                 wandb.log({
        #                     "energy": energy,
        #                     "eta": eta,
        #                     "gamma": gamma,
        #                     "generative_rule": rule,
        #                     "sweep_method": "bayesian"
        #                 })
        #             except Exception:
        #                 pass  # don’t fail the run if wandb isn’t active for this command
        #     except Exception as e:
        #         # keep going even if one experiment object is odd
        #         print(f"[warn] failed to serialize/log experiment {i}: {e}")


        # self._refresh_index_file()
        

    # view the experiments as a table and save as csv if you want
    def view_experiments():
        pass

    def _sort_experiments(self, experiments, variable_to_sort_by, get_names_only=False):

        def combine_dictionary_by_key(list_of_dictionaries, key_value_items):
            exp = {}
            for dictionary in list_of_dictionaries:
                for name in list(dictionary.keys()):
                    exp[name] = key_value_items[name][variable_to_sort_by]

            return exp

        experiment_names = list(experiments.keys())

        # keys = experiment name, values = experiment values of the given variable to sort by
        sorting_dict = {experiment_name:value for experiment_name, value in zip(experiment_names, [experiments[name][variable_to_sort_by] for name in experiment_names])}
        
        # # convert strings back to original 
        # sorting_dict = {experiment_name:eval(value) for experiment_name, value in sorting_dict.items()}

        # iterate to check types
        sorting_dict_numbers = {}
        sorting_dict_strings = {}
        sroting_dict_lists = {}
        for key, value in sorting_dict.items():
            if isinstance(value, int) or isinstance(value, float):
                sorting_dict_numbers[key] = value
            elif isinstance(value, str):
                sorting_dict_strings[key] = value
            else:
                sroting_dict_lists[key] = value

        
        # sort num, string dictionaries by values (sorted requires same datatype) - no point in sorting lists
        sorting_dict_numbers = dict(sorted(sorting_dict_numbers.items(), key=lambda item: item[1]))
        sorting_dict_strings = dict(sorted(sorting_dict_strings.items(), key=lambda item: item[1]))

        # create a new dictonary and add experiments based on the order of sorted_experiments
        # but with all the data included in experiments, rather than just the value used for sorting
        sorted_experiments = combine_dictionary_by_key([sorting_dict_numbers, sorting_dict_strings, sroting_dict_lists], experiments)
        
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

    """
    Deletes an experiment from the index file and removes the corresponding file from disk.
    """
    def delete_experiment(self, experiment_name, ask_first=True):
        if not self.save:
            warn('Parameter Save is False - no index file present so returning null')
            return

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

    def _is_similar_wording(self, variable_word, verbose=True):
        all_vars = self.variables_to_save

        char_frequency = {}
        for var in all_vars:
            letters_in_common = [character for character in variable_word if character in var]
            char_frequency[var] = len(letters_in_common) / len(var)

        char_frequency = dict(sorted(char_frequency.items(), key=lambda item: item[1]))
        most_likely_word = list(char_frequency.keys())[-1]

        if verbose:
            print(f'Did you mean {most_likely_word}?')
        
        return most_likely_word


    """
    Queries the index file for experiments matching a specified variable and value.
    Returns a list of experiment data files that match the criteria.
    """
    def query_experiments(self, value=None, by=None, limit=float('inf'), verbose=True):
        if not self.save:
            warn('Parameter Save is False - no index file present so returning null')
            return

        # get all searchable variables
        all_experiments = self.index_file['experiment_configs']
        if len(all_experiments) == 0:
            warn(f'No experiments saved in index file {self.index_file}')

        first_experiment = list(all_experiments.keys())[0]
        first_experiment_data = all_experiments[first_experiment]
        searchable_variables = list(first_experiment_data.keys())

        # make sure variable provided can be searched
        if by not in searchable_variables:
            print(f'Variable {by} not in searchable variables. Must be one of {searchable_variables}')
            self._is_similar_wording(by)
            return

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
            print(f'\nFound {len(experiment_data_to_return)} item(s) matching: {by} = {value}')
        
        return experiment_data_to_return
        
    """
    Opens experiments by their names and returns their data.
    If the name is 'test_config', it is skipped.
    """
    def open_experiments_by_name(self, experiment_names):
        if not self.save:
            warn('Parameter Save is False - no index file present so returning null')
            return
            
        if isinstance(experiment_names, str):
            experiment_names = [experiment_names]


        experiments_opened = []
        for name in experiment_names:
            if name == 'test_config':
                continue

            file_path = os.path.join(self.path, name)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f: # turned from "r" to "rb" to read binary files -> TODO: Check if this is correct
                    # experiments_opened.append(f)
                    experiments_opened.append(pickle.load(f)) # load the experiment data from the file -> TODO: Check if this is correct
            else:
                warn(f'File {name} could not be found in {file_path}. Do you have the right root folder specified?')
        
        # if len(experiments_opened) == 1:
        #     experiments_opened = experiments_opened[0]
        
        return experiments_opened