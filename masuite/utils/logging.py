from masuite.logging import base

class Logging():
    def __init__(self,
        logger: base.Logger,
        n_players: int,
        log_by_step: bool=False,
        log_every: bool=False,
        log_freq: int=10
    ) -> None:
        self.logger = logger
        self.n_players = n_players
        self.log_by_step = log_by_step
        self.log_every = log_every
        self.log_freq = log_freq

        # accumulating throughout experiment
        self.steps = 0
        self.episode = 0
        self.total_returns = [0.0 for _ in range(self.n_players)]

        # most recent episode
        self.episode_len = 0
        self.episode_returns = [0.0 for _ in range(self.n_players)]

        self.env_info = dict()
    
    def track_env_step(self, rews: list, done: bool, info: dict):
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
            if self.log_every or self.steps % self.log_freq == 0:
                print('logging')
                self.log_env_data()
        elif done:
            print('logging')
            self.log_env_data()

        if done:
            self.episode += 1
            self.episode_len = 0
            self.episode_returns = [0.0 for _ in range(self.n_players)]
    

    def log_env_data(self):
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