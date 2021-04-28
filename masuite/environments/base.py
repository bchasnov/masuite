import gym

class Environment(gym.Env):
    def __init__(self, mapping_seed=None, masuite_num_episodes=None):
        self.mapping_seed = mapping_seed
        self.masuite_num_episodes = masuite_num_episodes
        assert self.mapping_seed is not None
        assert self.n_players is not None
        assert self.masuite_num_episodes is not None

    def step(self, actions):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError