from masuite.logging import base

class Logging():
    def __init__(self,
        logger: base.Logger,
        n_players: int,
        log_by_step: bool=False,
        log_every: bool=False,
        log_freq: int=10,
        log_checkpoints: bool=True,
        checkpoint_freq: int=5000
    ) -> None:
        self.logger = logger
        self.n_players = n_players
        self.log_by_step = log_by_step
        self.log_every = log_every
        self.log_freq = log_freq
        self.log_checkpoints = log_checkpoints
        self.checkpoint_freq = checkpoint_freq

        # accumulating throughout experiment
        self.steps = 0
        self.episode = 0
        self.total_returns = [0.0 for _ in range(self.n_players)]
        self.checkpoints_logged = 0

        # most recent episode
        self.episode_len = 0
        self.episode_returns = [0.0 for _ in range(self.n_players)]

        self.env_info = dict()
    
    def log_timestep(self, rews: list, done: bool, info: dict):
        self.steps += 1
        self.episode_len += 1
        for player in range(self.n_players):
            self.episode_returns[player] += rews[player]
            self.total_returns[player] += rews[player]
        
        if info is not None:
            if self.env_info == {}:
                self.info = info
            else:
                for key, val in info.items():
                    self.info[key].append(val)

        if self.log_by_step:
            if done or self.log_every or self.steps % self.log_freq == 0:
                self._log_env_data()

        if done:
            self.episode += 1
            self.episode_len = 0
            self.episode_returns = [0.0 for _ in range(self.n_players)]
            self.env_info = {}


    def checkpoint_due(self)->bool:
        if not self.log_checkpoints:
            return False
        return (self.steps // self.checkpoint_freq) > self.checkpoints_logged
    

    def log_checkpoint(self, params):
        data = {}
        for i, param in enumerate(params):
            data[f'agent{i}']=param
        self.logger.write_checkpoint(data)
        self.checkpoints_logged += 1


    def _log_env_data(self):
        data = dict(
            steps=self.steps,
            episode=self.episode,
            total_returns=self.total_returns.copy(),
            episode_len=self.episode_len,
            episode_returns=self.episode_returns.copy()
        )
        if self.env_info != {}:
            for key, val in self.env_info.items():
                data[key] = val
        self.logger.write(data)
