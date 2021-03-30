import numpy as np
NUM_EPISODES = 50 # I chose this arbitrarily
NUM_PLAYERS = 2

_A = np.array([0.7])
_B = np.array([0.3])
_C = np.array([0.7])
_D = np.array([0.3])
SETTINGS = tuple({
    'mapping_seed': n,
    'A': _A,
    'B': _B,
    'C': _C,
    'D': _D} for n in range(20)
)
TAGS = ('basic',)