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
        self.batch_info = dict()
    
    
    def log_timestep(self, rews: list, done: bool, info: dict=None):
        self.steps += 1
        self.episode_len += 1
        for player in range(self.n_players):
            self.episode_returns[player] += rews[player]
            self.total_returns[player] += rews[player]
        
        if info is not None and info is not {}:
            if self.env_info == {}:
                self.env_info = info
            else:
                for key, val in info.items():
                    self.info[key].append(val)

        if self.log_by_step:
            if self.log_every or self.steps % self.log_freq == 0:
                self._log_env_data()
        elif done:
            if self.episode % self.log_freq == 0:
                self._log_env_data()

        if done:
            self.episode += 1
            self.episode_len = 0
            self.episode_returns = [0.0 for _ in range(self.n_players)]
            self.env_info = {}
    

    def track_batch_info(self, batch_info: dict):
        if self.batch_info == {}:
            self.batch_info = batch_info
        else:
            for key, value in batch_info.items():
                self.batch_info[key].append(value)


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
        data = {
            'steps': self.steps,
            'episode': self.episode,
            'episode_len': self.episode_len
        }
        for idx in range(self.n_players):
            data[f'agent{idx}_total_ret'] = self.total_returns[idx]
            data[f'agent{idx}_episode_ret'] = self.episode_returns[idx]

        if self.env_info != {}:
            for key, val in self.env_info.items():
                data[key] = val
        self.logger.write(data)