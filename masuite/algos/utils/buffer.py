from typing import NamedTuple
import numpy as np

class SingleBuffer:
    def __init__(
        self,
        max_batch_len: int
    ):
        self.max_batch_len= max_batch_len
        self._obs = []
        self._acts = []
        self._rews = []


    def append_reset(self, obs):
        print(self._obs)
        self._obs.append(obs.copy())
        print(self._obs)
    

    def append_obs(self, obs):
        self._obs.append(obs.copy())


    def append_timestep(self, acts, rews):
        """
        Append a single timestep to the trajectory buffer

        Args:
            obs: timestep state/observation information
            acts: timestep actions
            rews: timestep rewards
            done: whether or not the environment finished this time timestep
        """
        
        self._acts.append(acts)
        self._rews.append(rews)


    def compute_batch_info(self):
        #TODO: Discount factor
        ep_ret, ep_len = sum(self._rews), len(self._rews)
        self._rews = []
        return ep_ret, ep_len


    def drain(self):
        obs = self._obs
        acts = self._acts
        self._obs, self._acts = [], []
        return obs, acts 
    