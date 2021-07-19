import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

def conjugate_gradient_stac_pg(vec, params, b, vec2, x=None, nsteps=10, residual_tol=1e-18,
                       reg=0, device=torch.device('cpu')):
    if x is None:
        x = torch.zeros(b.shape, device=device)

    _Ax = autograd.grad(vec, params, grad_outputs=x, retain_graph=True)
    Ax = torch.cat([g.contiguous().view(-1) for g in _Ax])
    Ax += vec * torch.dot(vec2, x)  # special conjugate gradient just for stackelberg policy gradient
    Ax += reg * x

    r = b.clone().detach() - Ax
    p = r.clone().detach()
    rsold = torch.dot(r.view(-1), r.view(-1))

    for itr in range(nsteps):
        _Ap = autograd.grad(vec, params, grad_outputs=p, retain_graph=True)
        Ap = torch.cat([g.contiguous().view(-1) for g in _Ap])
        Ap += reg * p

        alpha = rsold / torch.dot(p.view(-1), Ap.view(-1))
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Ap)
        rsnew = torch.dot(r.view(-1), r.view(-1))
        if rsnew < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, itr + 1

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


def stac_train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net1 = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
    logits_net2 = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy1(obs):
        logits = logits_net1(obs)
        return Categorical(logits=logits)
    def get_policy2(obs):
        logits = logits_net2(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action1(obs):
        return get_policy1(obs).sample().item()
    def get_action2(obs):
        return get_policy1(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    # here consider player 1 is the leader and player 2 is the follower
    # player 1 is maximizing the total reward and player 2 minimizes that, so it is zero-sum
    def compute_loss1(obs, act1, weights):  # for D1f1
        logp1 = get_policy1(obs).log_prob(act1)
        return -(logp1 * weights).mean()

    def compute_loss2(obs, act2, weights):  # for D2f2 and -D2f1
        logp2 = get_policy2(obs).log_prob(act2)
        return (logp2 * weights).mean()

    def compute_loss12(obs, act1, act2, weights):  # for D12f2
        logp1 = get_policy1(obs).log_prob(act1)
        logp2 = get_policy2(obs).log_prob(act2)
        return (logp1 * logp2 * weights).mean()

    def compute_loss_p(obs, act2):  # for part of D22f2
        logp2 = get_policy2(obs).log_prob(act2)
        return (logp2).mean()

    # make optimizer
    optimizer1 = Adam(logits_net1.parameters(), lr=lr)
    optimizer2 = Adam(logits_net2.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts1 = []         # for actions
        batch_acts2 = []
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act1 = get_action1(torch.as_tensor(obs, dtype=torch.float32))
            act2 = get_action2(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act1, act2)

            # save action, reward
            batch_acts1.append(act1)
            batch_acts2.append(act2)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        p1, p2 = list(logits_net1.parameters()), list(logits_net2.parameters())

        f1 = compute_loss1(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act1=torch.as_tensor(batch_acts1, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        f2 = compute_loss2(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act2=torch.as_tensor(batch_acts2, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        D1f1 = autograd.grad(f1, p1, create_graph=True)
        D1f1_vec = torch.cat([g.contiguous().view(-1) for g in D1f1])
        D2f2 = autograd.grad(f2, p2, create_graph=True)
        D2f2_vec = torch.cat([g.contiguous().view(-1) for g in D2f2])
        D2f1_vec = -D2f2_vec.clone()  # f1 = -f2

        f_p = compute_loss_p(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act2=torch.as_tensor(batch_acts2, dtype=torch.int32)
                                  )
        D2p = autograd.grad(f_p, p2, create_graph=True)
        D2p_vec = torch.cat([g.contiguous().view(-1) for g in D2p])

        x, _ = conjugate_gradient_stac_pg(D2f2_vec, p2, D2f1_vec.detach(), D2p_vec.detach(), reg=reg, device=device)  # D22f2^-1 * D2f1

        f2_surro = compute_loss12(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act1=torch.as_tensor(batch_acts1, dtype=torch.int32),
                                  act2=torch.as_tensor(batch_acts2, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        D2f2_surro = autograd.grad(f2_surro, p2, create_graph=True)
        D2f2_surro_vec = torch.cat([g.contiguous().view(-1) for g in D2f2_surro])

        _Avec = autograd.grad(D2f2_surro_vec, p1, x, retain_graph=True, allow_unused=True)
        grad_imp = torch.cat(
            [g.contiguous().view(-1) if g is not None else torch.Tensor([0]).to(device) for g in _Avec])
        grad_stac = D1f1_vec.detach() - grad_imp  # D1f1 - D12f2 * D22f2^-1 * D2f1

        # take a single policy gradient update step
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        def backward(params, grad):
            index = 0
            for p in params:
                p.grad.add_(grad[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            if index != grad.numel():
                raise ValueError('gradient size mismatch')

        backward(p1, grad_stac)
        backward(p2, D2f2_vec.detach())

        optimizer1.step()
        optimizer2.step()
        
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)