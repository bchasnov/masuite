from typing import NamedTuple
import numpy as np

class SingleBuffer:
    def __init__(
        self,
        obs_dim: list,
        act_dim: int,
        max_batch_len: int
    ):
        # self._obs = np.zeros(shape=(max_batch_len+1, *obs_dim))
        # if act_dim == 1:
        #     self._acts = np.zeros(shape=(max_batch_len))
        # else:
        #     self._acts = np.zeros(shape=(max_batch_len, act_dim))
        # self._rews = np.zeros(shape=(max_batch_len))
        self.max_batch_len= max_batch_len
        self._obs = []
        self._acts = []
        self._rews = []
        self._needs_reset = True


    def append_reset(self, obs):
        self._obs.append(obs)


    def append_timestep(self, obs, acts, rews):
        """
        Append a single timestep to the trajectory buffer

        Args:
            obs: timestep state/observation information
            acts: timestep actions
            rews: timestep rewards
            done: whether or not the environment finished this time timestep
        """
        if self._needs_reset:
            self._obs, self._acts, self._rews = [], [], []
            self._needs_reset = False
        
        # self._obs[self._curr_len+1] = obs
        # self._acts[self._curr_len+1] = acts
        # self._rews[self._curr_len+1] = rews
        # self._curr_len += 1
        self._obs.append(obs)
        self._acts.append(acts)
        self._rews.append(rews)


    def compute_batch_info(self):
        #TODO: Discount factor
        ep_ret = sum(self._rews)
        ep_len = len(self._rews)
        self._rews = []
        return ep_ret, ep_len


    def drain(self):
        obs = np.array(self._obs)
        acts = np.array(self._acts)
        self._needs_reset = True
        return obs, acts 
    