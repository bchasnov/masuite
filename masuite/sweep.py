from typing import Any, Dict, Mapping, Tuple

from masuite.experiments.cartpole import sweep as cartpole_sweep
from masuite.experiments.cartpole2p import sweep as cartpole2p_sweep
from masuite.experiments.smallsoccer import sweep as smallsoccer_sweep

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
        # _EPISODES[masuite_id] = experiment_package.NUM_EPISODES
    
    # Add masuite_ids to corresponding tag sweeps
    _SWEEP.extend(results)
    return tuple(results)


# masuite_ids broken down by environment
CARTPOLE = _parse_experiment_sweep(cartpole_sweep)
CARTPOLE2P = _parse_experiment_sweep(cartpole2p_sweep)
SOCCER_SMALL = _parse_experiment_sweep(smallsoccer_sweep)

# mapping from masuite id to keyword arguments for the corresponding env
SETTINGS: Mapping[MASuiteId, EnvKWargs] = frozendict.frozendict(**_SETTINGS)

# Tuple containing all masuite_ids. Used for hyperparameter sweeps
SWEEP: Tuple[MASuiteId, ...] = tuple(_SWEEP)

# # mapping from masuite_id to masuite_num_episodes = how many episodes to run
# EPISODES: Mapping[MASuiteId, int] = frozendict.frozendict(**_EPISODES)
