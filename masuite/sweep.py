from typing import Any, Dict, Mapping, Tuple

from masuite.experiments.quadratic_2p_simgrad import sweep as quadratic_2p_simgrad_sweep
from masuite.experiments.cartpole_simplepg import sweep as cartpole_simplepg_sweep
from masuite.experiments.cartpole2p_simplepg import sweep as cartpole2p_simplepg_sweep
from masuite.experiments.soccer_simplepg import sweep as soccer_simplepg_sweep

import frozendict

# Common type aliases
MASuiteId = str
Tag = str
EnvKWargs = Dict[str, Any]

SEP = '/'

_SETTINGS = {}
_SWEEP = []
_EPISODES = {}

def _parse_experiment_sweep(experiment_package)->Tuple[MASuiteId,...]:
    """Returns the masuite_ids for each experiment package"""
    results = []
    # package.__name is something like 'masuite.experiements.quadratic_2p_simgrad.sweep
    experiment_name = experiment_package.__name__.split('.')[-2]

    # construct masuite_ids for each setting defined by the experiment
    for i, setting in enumerate(experiment_package.SETTINGS):
        masuite_id = f'{experiment_name}{SEP}{i}'
        results.append(masuite_id)
        _SETTINGS[masuite_id] = setting
        _EPISODES[masuite_id] = experiment_package.NUM_EPISODES
    
    # Add masuite_ids to corresponding tag sweeps
    _SWEEP.extend(results)
    return tuple(results)


# masuite_ids broken down by environment
QUADRATIC_2P_SIMGRAD = _parse_experiment_sweep(quadratic_2p_simgrad_sweep)
CARTPOLE_SIMPLEPG = _parse_experiment_sweep(cartpole_simplepg_sweep)
CARTPOLE2P_SIMPLEPG = _parse_experiment_sweep(cartpole2p_simplepg_sweep)
SOCCER_SIMPLEPG = _parse_experiment_sweep(soccer_simplepg_sweep)

# mapping from masuite id to keyword arguments for the corresponding env
SETTINGS: Mapping[MASuiteId, EnvKWargs] = frozendict.frozendict(**_SETTINGS)

# Tuple containing all masuite_ids. Used for hyperparameter sweeps
SWEEP: Tuple[MASuiteId, ...] = tuple(_SWEEP)

# mapping from masuite_id to masuite_num_episodes = how many episodes to run
EPISODES: Mapping[MASuiteId, int] = frozendict.frozendict(**_EPISODES)