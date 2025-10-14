import os

import numpy as np
import random

from kit.data import DD

def get_hyp_params(yaml_file_path):
    """ returns a DD with the command and the 
    populated hyperparameters from the YAML file

    :param yaml_file_path: path to the YAML file specifying the hyperparameters
        and how to sample them
    :return: DD with the command and the 
        populated hyperparameters
    """

    if not os.path.exists(yaml_file_path):
        raise Exception(f"YAML file {yaml_file_path} does not exist")
    
    hyp_params = DD.from_yaml(yaml_file_path, evaluate=False)

    d_cmd = DD.from_dict({'COMMAND': hyp_params.COMMAND})
    for hp in hyp_params.HYPERPARAMETERS:
        kind, hp_type = hyp_params.HYPERPARAMETERS[hp].KIND.split(":")
        value = None
        if kind == "fixed":
            value = hyp_params.HYPERPARAMETERS[hp].VALUE
        elif kind == "uniform":
            value = random.uniform(
                float(hyp_params.HYPERPARAMETERS[hp].MIN), 
                float(hyp_params.HYPERPARAMETERS[hp].MAX)
            )
        elif kind == "log_uniform":
            _min = np.log(float(hyp_params.HYPERPARAMETERS[hp].MIN))
            _max = np.log(float(hyp_params.HYPERPARAMETERS[hp].MAX))
            value = np.exp(random.uniform(_min, _max))
        elif kind == "env":
            value = os.environ[hyp_params.HYPERPARAMETERS[hp].NAME]
            if "EVAL" in hyp_params.HYPERPARAMETERS[hp]:
                func = eval(hyp_params.HYPERPARAMETERS[hp].EVAL)
                value = func(value)

        if value is None:
            raise Exception(f"Hyperparameter {hp} not well defined in YAML file {yaml_file_path}")

        value = eval(f"{hp_type}(value)")

        d_cmd[hp] = value

    return d_cmd
