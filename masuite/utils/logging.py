from masuite.logging import base

class EpochLogging:
    def __init__(self,
        logger: base.Logger,
        n_players: int,
        log_freq: int=1,
        log_checkpoints: bool=False,
        checkpoint_freq: int=5
    ) -> None:
        self.logger = logger
        self.n_players = n_players
        self.log_freq = log_freq
        self.log_checkpoints = log_checkpoints
        self.checkpoint_freq = checkpoint_freq

        # accumulating throughout experiment
        self.epoch = 0


    def track_epoch(self, epoch_info: dict) -> None:
        if self.epoch % self.log_freq == 0:
            data = {}
            for idx in range(self.n_players):
                for key, value in epoch_info.items():
                    data[f'agent{idx}_{key}'] = value[idx]
            self.logger.write(data)
    

    def checkpoint_due(self):
        return self.log_checkpoints and self.epoch % self.checkpoint_freq == 0

