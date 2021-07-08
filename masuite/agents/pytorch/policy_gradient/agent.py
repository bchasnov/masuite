import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity, seed=None):
    """Initialize a feed-forward neural network
    
    Keyword arguments:
    sizes -- sizes for the layers of the neural network
    activation -- input activation function for the neural network
    output activation -- output activation function for the neural network
    seed -- to seed the intial state of the neural network
    
    returns -- torch.nn.Sequential initialized neural network"""
    torch.manual_seed(0)
    if seed is not None:
        torch.manual_seed(seed)
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
        lr: float=1e-2,
        seed: int=None,
        optim=Adam
    ):
        if not isinstance(env_dim, list):
            env_dim = [env_dim]
        if not isinstance(n_acts, list):
            n_acts = [n_acts]
        self.logits_net = mlp(sizes=env_dim+hidden_sizes+n_acts, seed=seed)
        self.lr = lr
        self.optim = optim(self.logits_net.parameters(), lr=lr)


    def _get_policy(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    

    def _get_params(self):
        return self.logits_net.parameters()
    

    def _zero_grad(self):
        for p in self.logits_net.parameters():
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()
    

    def select_action(self, obs):
        """Choose an action given the agents current action policy and the
        timestep's environment observations
        
        Keyword arguments:
        obs -- torch.Tensor observations from environment for this timestep
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs).float()
        return self._get_policy(obs).sample().item()
        

    def update(self, grads):
        """Update agent's neural network parameters given grads
        
        Keyword arguments:
        grads -- tuple of torch.Tensor containing the gradients to apply this
        step
        """
        self._zero_grad()
        for param, grad in zip(self.logits_net.parameters(), grads):
            param.grad = grad
        self.optim.step()


def default_agent(env_dim, act_dim):
    return PGAgent(env_dim, act_dim)
