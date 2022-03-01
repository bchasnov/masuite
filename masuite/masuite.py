from typing import Any, Mapping, Tuple

from masuite import sweep
from masuite.environments import base

# experiments
from masuite.experiments.cartpole import cartpole
from masuite.experiments.cartpole2p import cartpole2p
from masuite.experiments.smallsoccer import smallsoccer
from masuite.experiments.soccer import soccer
from masuite.experiments.gridhunter import gridhunter

# logging
from masuite.utils.logging import EpochLogging
from masuite.logging.csv_logging import CSVLogger
from masuite.logging.terminal_logging import TerminalLogger


# Mapping from experiment name to environment constructor or load function.
# Each constructor or load function accepts keywork arguments as defined in
# each experiment's sweep.py file
EXPERIMENT_NAME_TO_ENVIRONMENT = dict(
    cartpole=cartpole.load,
    cartpole2p=cartpole2p.load,
    smallsoccer=smallsoccer.load,
    soccer=soccer.load,
    gridhunter=gridhunter.load,
)


def unpack_masuite_id(masuite_id: str)->Tuple[str, int]:
    """Returns the experiment name and setting index given an masuite_id"""
    parts = masuite_id.split(sweep.SEP)
    assert len(parts) == 2
    return parts[0], int(parts[1]) # (experiment_name, setting_index)


def load(exp_name: str, kwargs: Mapping[str, Any])->base.Environment:
    """Load and initialize an environment corresponding to exp_name
    
    Keyword arguments:
    exp_name -- str the name of the experiment
    
    returns -- initialized masuite.Environment"""
    return EXPERIMENT_NAME_TO_ENVIRONMENT[exp_name](**kwargs)


def load_from_id(masuite_id: str)->base.Environment:
    """Parse masuite_id and load the corresponding environment
    
    Keyword arguments:
    masuite_id -- str masuite_id corresponding to the experiment to be run
    
    returns -- initialized masuite.Environment
    """
    kwargs = sweep.SETTINGS[masuite_id]
    exp_name, _ = unpack_masuite_id(masuite_id)
    env = load(exp_name, kwargs)
    print(f'Loaded masuite_id: {masuite_id}.')
    return env


def init_logging(
    filename: str,
    n_players: int,
    mode: str,
    save_path: str,
    overwrite: bool,
    log_freq: int,
    log_checkpoints: bool,
    checkpoint_freq: int,
    params: dict=None
):
    if mode == 'csv':
        log_class = CSVLogger
    elif mode == 'terminal':
        log_class = TerminalLogger
    else:
        raise ValueError('Invalid terminal logging mode: \'{mode}\'')
        
    logger = log_class(
        filename=filename,
        results_dir=save_path,
        overwrite=overwrite,
        log_checkpoints=log_checkpoints,
        params=params
    )

    logging_instance = EpochLogging(
        logger=logger,
        n_players=n_players,
        log_freq=log_freq,
        log_checkpoints=log_checkpoints,
        checkpoint_freq=checkpoint_freq
    )
    
    return logging_instance
