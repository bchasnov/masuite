import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete
import numpy as np

class QuadraticCritic(nn.Module):
    """ represents 
         V(x) = x^T P x 
    """
    def __init__(self, obs_dim, P_init=None):
        # super(QuadraticCritic, self).__init__()
        super().__init__()
        self.P = torch.nn.Parameter(torch.eye(obs_dim) if P_init is None else P_init)

    def forward(self, obs):
        return obs @ self.P @ obs.t()

class LinearCategoricalActor(nn.Module):
    """ represents linear MDP policy
          act is sampled from y 
         where y=L@obs where x, u are on the simplex and L is row stochastic .
    """
    def __init__(self, obs_dim, act_dim, L_init=None):
        super().__init__()
        self.L = torch.nn.Parameter(torch.zeros((act_dim, obs_dim)) if L_init is None else L_init)
        
    def _distribution(self, obs):
        logits = self.L@obs
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions. (taken from spinningup so we don't need to
        # inherit their Actor class)
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class LinearGaussianActor(nn.Module):
    """ represents linear state feedback
     act = K@obs + w
     where w ~ N(0, std)
    """
    def __init__(self, obs_dim, act_dim, noise_std, K_init=None):
        super().__init__()
        #self.log_std = torch.nn.Parameter(torch.as_tensor(np.log(noise_std)))
        self.log_std = torch.as_tensor(np.log(noise_std))
        self.K = torch.nn.Parameter(torch.zeros((act_dim, obs_dim)) if K_init is None else K_init)

    def _distribution(self, obs):
        mu = self.K@obs.t()
        return Normal(mu, torch.exp(self.log_std))
        #return Normal(mu, torch.exp(self.log_std))

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions. (taken from spinningup so we don't need to
        # inherit their Actor class)
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class LQActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, 
                 activation=nn.Identity, K_init=None, P_init=None, noise_std=1e-1):
        # super(LQActorCritic, self).__init__()
        super().__init__()
        obs_dim = obs_space.shape[0]

        if isinstance(act_space, Box):
            self.pi = LinearGaussianActor(obs_dim, act_space.shape[1], K_init=K_init, noise_std=noise_std)
        elif isinstance(action_space, Discrete):
            self.pi = LinearCategoricalActor(obs_dim, actions_space.n)
        
        self.v = QuadraticCritic(obs_dim, P_init=P_init)
    
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v.forward(obs)
        return a.numpy().flatten(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]