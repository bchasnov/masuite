from typing import Any, Mapping, Tuple

from masuite import sweep
from masuite.environments import base
from masuite.experiments.quadratic_2p_simgrad import quadratic_2p_simgrad
from masuite.experiments.cartpole_simplepg import cartpole_simplepg
from masuite.experiments.cartpole2p_simplepg import cartpole2p_simplepg

from masuite.logging import csv_logging
from masuite.logging import terminal_logging


# Mapping from experiment name to environment constructor or load function.
# Each constructor or load function accepts keywork arguments as defined in
# each experiment's sweep.py file
EXPERIMENT_NAME_TO_ENVIRONMENT = dict(
    quadratic_2p_simgrad=quadratic_2p_simgrad.load,
    cartpole_simplepg=cartpole_simplepg.load,
    cartpole2p_simplepg=cartpole2p_simplepg.load
)


def unpack_masuite_id(masuite_id: str)->Tuple[str, int]:
    """Returns the experiment name and setting index given an masuite_id"""
    parts = masuite_id.split(sweep.SEP)
    assert len(parts) == 2
    return parts[0], int(parts[1]) # (experiment_name, setting_index)


def load(exp_name: str, kwargs: Mapping[str, Any])->base.Environment:
    return EXPERIMENT_NAME_TO_ENVIRONMENT[exp_name](**kwargs)



def load_from_id(masuite_id: str)->base.Environment:
    kwargs = sweep.SETTINGS[masuite_id]
    exp_name, _ = unpack_masuite_id(masuite_id)
    env = load(exp_name, kwargs)
    print(f'Loaded masuite_id: {masuite_id}.')
    return env

def load_and_record(masuite_id: str,
                    save_path: str,
                    batch_size: int,
                    logging_mode: str='csv',
                    overwrite: bool=False,)->base.Environment:
    if logging_mode == 'csv':
        return load_and_record_to_csv(masuite_id, save_path, batch_size, overwrite)
    elif logging_mode == 'terminal':
        return load_and_record_to_terminal(masuite_id)
    else:
        raise NotImplementedError


def load_and_record_to_csv(masuite_id: str,
                           results_dir: str,
                           batch_size,
                           overwrite: bool=False)->base.Environment:
    raw_env = load_from_id(masuite_id)
    print(f'Logging results to CSV file for each masuite_id in {results_dir}.')
    return csv_logging.wrap_environment(
        env=raw_env,
        batch_size=batch_size,
        masuite_id=masuite_id,
        results_dir=results_dir,
        overwrite=overwrite
    )


def load_and_record_to_terminal(masuite_id: str)->base.Environment:
    raw_env = load_from_id(masuite_id)
    print('Logging results to terminal.')
    return terminal_logging.wrap_environment(raw_env)
