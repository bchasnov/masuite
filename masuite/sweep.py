from typing import Any, Dict, Mapping, Tuple

from masuite.experiments.quadratic_2p_simgrad import sweep as quadratic_2p_simgrad_sweep
from masuite.experiments.cartpole_simplepg import sweep as cartpole_simplepg_sweep

import frozendict

# Common type aliases
MASuiteId = str
Tag = str
EnvKWargs = Dict[str, Any]

SEP = '/'

IGNORE_FOR_TESTING = ('_noise', '_scale')

_SETTINGS = {}
_SWEEP = []
_TAGS = {}
_TESTING = []
_EPISODES = {}

def _parse_sweep(experiment_package)->Tuple[MASuiteId,...]:
    """Returns the masuite_ids for each experiment package"""
    results = []
    # package.__name is something like 'masuite.experiements.quadratic_2p_simgrad.sweep
    experiment_name = experiment_package.__name__.split('.')[-2]
    eligible_for_test_sweep = not any(experiment_name.endswith(s) for s in IGNORE_FOR_TESTING)

    # construct masuite_ids for each setting defined by the experiment
    for i, setting in enumerate(experiment_package.SETTINGS):
        masuite_id = f'{experiment_name}{SEP}{i}'
        if i == 0 and eligible_for_test_sweep:
            # For each environment, add one masuite_id to the TESTING sweep
            _TESTING.append(masuite_id)
        results.append(masuite_id)
        _SETTINGS[masuite_id] = setting
        _EPISODES[masuite_id] = experiment_package.NUM_EPISODES
    
    # Add masuite_ids to corresponding tag sweeps
    for tag in experiment_package.TAGS:
        if tag not in _TAGS:
            _TAGS[tag] = []
        _TAGS[tag].extend(results)
    _SWEEP.extend(results)
    return tuple(results)


# masuite_ids broken down by environment
QUADRATIC_2P_SIMGRAD = _parse_sweep(quadratic_2p_simgrad_sweep)
CARTPOLE_SIMPLEPG = _parse_sweep(cartpole_simplepg_sweep)

# mapping from masuite id to keyword arguments for the corresponding env
SETTINGS: Mapping[MASuiteId, EnvKWargs] = frozendict.frozendict(**_SETTINGS)

# Tuple containing all masuite_ids. Used for hyperparameter sweeps
SWEEP: Tuple[MASuiteId, ...] = tuple(_SWEEP)

# mapping from tag (e.g. 'memory') to experiment masuite_ids with that tag.
# this can be used to run sweeps on all tasks only of a particular tag, by using
# TAGS['basic'] or TAGS['scale']
TAGS: Mapping[Tag, Tuple[MASuiteId, ...]] = frozendict.frozendict(
    **{k: tuple(v) for k, v in _TAGS.items()}
)

# tuple containing a representative subset masuite_ids used for agent tests
TESTING: Tuple[MASuiteId, ...] = tuple(_TESTING)

# mapping from masuite_id to masuite_num_episodes = how many episodes to run
EPISODES: Mapping[MASuiteId, int] = frozendict.frozendict(**_EPISODES)