import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

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
        act_dim: int,
        hidden_sizes: list(int)=[32],
        lr: float=1e-2
    ):
        self.logits_net = mlp(sizes=[env_dim]+hidden_sizes+act_dim)
        self.lr = lr

    def _get_policy(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _zero_grad(self):
        for p in self.logits_net.parameters():
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()
    
    def _get_params(self):
        return self.logits_net.parameters()
    
    def select_action(self, obs):
        return self._get_policy(obs).sample().item()
        
    def update(self, grad):
        self._zero_grad()
        idx = 0
        for p in self.logits_net.parameters():
            p.data.add_(-self.lr * grad[index:index+p.numel()].reshape(p.shape))
            index += p.numel()
        if index != grad.numel():
            raise ValueError('Gradient size mismatch')

class PolicyGradient():
    def __init__(self,
        env, #algos
        n_players, #algos
        hidden_sizes=[32],
        lr=1e-2, #algos
        epochs=50, #algos
        batch_size=5000, #algos
        render=False,
        optim=Adam #algos
    ):
        self.n_players = n_players
        if isinstance(lr, list):
            self.lr = lr
        else:
            self.lr = [lr for _ in range(self.n_players)]
        self.epochs = epochs
        self.batch_size = batch_size
        self.render = render
        #TODO: How to init env? Maybe instead pass an instance of the env
        self.env = gym.make(env_name)
        self.obs_dim = env.observation_space.shape[0]
        self.n_acts = env.action_space.n
        self.logits_nets, self.optims = [], []
        for idx in range(self.n_players):
            if isinstance(self.lr, list):
                player_lr = self.lr[idx]
            else:
                player_lr = self.lr
            player_logits_net = mlp(sizes=[self.obs_dim]+hidden_sizes+self.n_acts)
            player_optim = optim(player_logits_net.parameters(), lr=lr)
            self.logits_nets.append(player_logits_net)
            self.optims.append(player_optim)
    

    def _get_policy(self, obs, player):
        #? Does this need to change for continuous action spaces like AC?
        logits = self.logits_nets[player](obs)
        return Categorical(logits=logits)

    
    def _get_action(self, obs, player):
        return self._get_policy(obs, player).sample().item()
    

    def _compute_loss(self, act, weights, player):
        player_acts = [acts[i][player] for i in range(len(act))]
        logp = self._get_policy(obs, player).log_prob(player_acts)
        return -(logp*weights).mean()
    

    def _train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for mtcheasuring episode lengths

        # reset episode-specific variables
        obs = self.env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        for idx in range(self.batch_size):

            # rendering
            if (not finished_rendering_this_epoch) and self.render:
                self.env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            #TODO: self.select_action(obs)
            acts = []
            for player in range(self.n_players):
                act = self._get_action(torch.as_tensor(obs, dtype=torch.float32), player)
                acts.append(act)
            obs, rews, done, _ = self.env.step(acts)


            # save action, reward
            batch_acts.append(acts)
            ep_rews.append(rews)

            if done or idx == self.batch_size-1:
                # if episode is over, record info about episode
                #TODO: calculate for each player
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = self.env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) >= batch_size:
                    break

        # take a single policy gradient update step
        batch_losses = []
        for optim in self.optims:
            optim.zero_grad()
            batch_loss = self._compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.float32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
            batch_losses.append(batch_loss)
            batch_loss.backward()
            optim.step()
        return batch_losses, batch_rets, batch_lens

    
    def _train(self):
        # training loop
        for i in range(epochs):
            batch_loss, batch_rets, batch_lens = train_one_epoch()
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                    (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


    def select_action(self):
        pass


    def update(self, action):
        pass

def default_agent(env, lr=1e-2):
    return PolicyGradient(
        env=env,
        n_players=env.n_players,
        lr=lr
    )


if __name__ == '__main__':
    from masuite.environments.cartpole import CartPoleEnv
    