from typing import NamedTuple
import numpy as np

class Buffer:
    def __init__(
        self,
        obs_dim: list,
        act_dim: list,
        n_players: int,
        max_batch_len: int
    ):
        self._obs = np.zeros(shape=(max_batch_len+1, *obs_dim))
        self._acts = np.zeros(shape=(max_batch_len, n_players, *act_dim))
        self._rews = np.zeros(shape=(max_batch_len, n_players))
        self._batch_rets = np.zeros(shape=(max_batch_len, n_players))
        self._batch_lens = np.zeros(shape=max_batch_len)
        self.n_players = n_players
        self.max_batch_len= max_batch_len
        self._curr_len = 0
        self._needs_reset = True


    def append_reset(self, obs):
        self._obs[self._curr_len] = obs


    def append_timestep(self, obs, acts, rews):
        """
        Append a single timestep to the trajectory buffer

        Args:
            obs: timestep state/observation information
            acts: timestep actions
            rews: timestep rewards
            done: whether or not the environment finished this time timestep
        """
        if self.full():
            raise ValueError('Cannot append; buffer is full')
        
        self._obs[self._curr_len+1] = obs
        self._acts[self._curr_len+1] = acts
        self._rews[self._curr_len+1] = rews
        self._curr_len += 1


    def compute_batch_info(self):
        ep_ret = []
        if self.n_players == 1:
            ep_ret.append(sum(self._rews))
        else:
            for idx in range(self.n_players):
                ep_ret.append(sum(self._rews[:, idx]))
        ep_len = len(self._rews)
        return ep_ret, ep_len


    def drain(self):
        if self.empty():
            raise ValueError('Cannot drain; buffer is empty')
        obs = self._obs[:self._curr_len]
        acts = self._acts[:self._curr_len]
        self._curr_len = 0
        self._needs_reset = True
        return obs, acts 
    
    
    def empty(self)->bool:
        return self._curr_len == 0


    def full(self)->bool:
        return self._curr_len == self.max_batch_len