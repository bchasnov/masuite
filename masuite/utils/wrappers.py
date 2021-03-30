from masuite import environments
from masuite.logging import base
import gym

STANDARD_KEYS = frozenset([
    'steps',
    'episode',
    'total_return',
    'episode_len',
    'episode_return'
])


class Logging(environments.Environment):
    def __init__(self, env: environments.Environment,
        logger: base.Logger,
        log_by_step: bool=False,
        log_every: bool=False,
        log_freq: int=10):
    
        """Initializes the logging wrapper.
        Args:
        env: Environment to wrap.
        logger: An object that records a row of data. This must have a `write`
        method that accepts a dictionary mapping from column name to value.
        log_by_step: Whether to log based on step or episode count (default).
        log_every: Forces logging at each step or episode, e.g. for debugging.
        log_freq: Frequency to output to the log file (in steps)
        """
        self.env = env
        self.logger = logger
        self.log_by_step = log_by_step
        self.log_every = log_every

        # accumulating throughout experiment
        self.steps = 0
        self.episode = 0
        self.total_returns = [0.0 for _ in range(env.n_players)]

        # most recent episode
        self.episode_len = 0
        self.episode_return = [0.0 for _ in range(env.n_players)]
        
    def flush(self):
        if hasattr(self.logger, 'flush'):
            self.logger.flush()
    
    def reset(self):
        obs = self.env.reset()
        return obs
    
    def step(self, actions):
        obs, rews, done, info = self.env.step(actions)
        return obs, rews, done, info
    
    def track(self, obs, rews=None, done=None, info=None):
        if rews is not None:
            assert len(rews) == self.env.n_players, 'must be 1 reward for each player'
        
        if rews is not None:
            self.steps += 1
            self.episode_len += 1
        
        if done:
            self.episode += 1
        
        for player in range(self.env.n_players):
            self.episode_return[player] += rews[player]
            self.total_return[player] += rews[player]
        
        if self.log_by_step:
            if self.log_every or self.steps % self.log_freq == 0:
                self.log_masuite_data()
        elif done:
            if self.log_every or self.steps % self.log_freq == 0:
                self.log_masuite_data()
        
        if done:
            self.episode_len = 0
            self.episode_return = [0.0 for _ in range(self.env.n_players)]

        if self.episode == self.env.masuite_num_episodes:
            self.flush()
        
    def __getattr__(self, attr):
        return getattr(self.env, attr)
    
    def log_masuite_data(self):
        info = dict(
            steps=self.steps,
            epidsode=self.episode,
            total_return=self.total_return,
            episode_len=self.episode_len,
            episode_return=self.episode_return
        )

        #data.update(self.env.masuite_info())
        self.logger.write(data)
