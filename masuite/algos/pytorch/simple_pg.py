from torch.optim import Adam

class SimplePG:
    def __init__(
        env, #? env_name?
        lr,
        epochs,
        batch_size,
        render,
        optim=Adam,
        agent
    ):
    self.env = gym.make(env)
    self.lr = lr
    