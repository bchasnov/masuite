import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class PGAgent:
    def __init__(self,
        env_dim: int,
        n_acts: int,
        hidden_sizes: list=[32],
        lr: float=1e-2
    ):
        if not isinstance(env_dim, list):
            env_dim = [env_dim]
        if not isinstance(n_acts, list):
            n_acts = [n_acts]
        self.logits_net = mlp(sizes=env_dim+hidden_sizes+n_acts)
        # print(env_dim+hidden_sizes+n_acts)
        self.lr = lr


    def _get_policy(self, obs):
        logits = self.logits_net(obs)
        # print('logits: ', logits)
        return Categorical(logits=logits)


    def _zero_grad(self):
        for p in self.logits_net.parameters():
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()
    

    def _get_params(self):
        return self.logits_net.parameters()
    

    def select_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs).float()
        return self._get_policy(obs).sample().item()
        

    def update(self, grad):
        self._zero_grad()
        for p, g in zip(self.logits_net.parameters(), grad):
            p.data.add_(-self.lr * g)

def default_agent(env_dim, act_dim):
    return PGAgent(env_dim, act_dim)
