import torch
import torch.autograd as autograd
import numpy as np
from masuite.algos.utils.buffer import SingleBuffer

class SimplePG:
    def __init__(self,
        agents,
        shared_state: bool,
        n_players: int,
        batch_size: int
    ):
        self.agents = agents
        self.shared_state = shared_state
        self.n_players = n_players
        self.batch_size = batch_size
        self.buffers = [SingleBuffer(
            max_batch_len=batch_size
        ) for _ in range(n_players)]
        self._reset_batch_info()
    

    def _reset_batch_info(self):
        """Reset the tracked batch info"""
        self.batch_rets, self.batch_lens, self.batch_weights = [], [], []
        for _ in range(self.n_players):
            self.batch_rets.append([])
            self.batch_lens.append([])
            self.batch_weights.append([])

    
    def _compute_loss(self, obs, act, weights, agent):
        """Compute the log-probability loss for the current batch.
        
        Keyword arguments:
        obs -- torch.tensor containing the batch observations
        corresponding to the current agent
        acts -- torch.tensor containing the agent's actions for the current
        batch
        rews -- torch.tensor containing the agent's rewards for the current
        batch
        agent -- masuite.agent instance which agent to compute the loss for

        returns -- float agent's log-prob loss for current batch
        """
        logp = agent._get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    

    def _step(self, obs, acts):
        """Compute the batch loss and take a single gradient descent
        optimization step for each agent.

        Keyword arguments:
        obs -- list of batch observations
        acts -- list of batch actions taken
        """
        info = dict(loss=[])
        grads = []
        for idx in range(self.n_players):
            agent = self.agents[idx]
            obs_ = torch.as_tensor(obs[idx], dtype=torch.float32)
            acts_ = torch.as_tensor(acts[idx], dtype=torch.int32)
            weights_ = torch.as_tensor(self.batch_weights[idx], dtype=torch.float32)
            loss = self._compute_loss(obs_, acts_, weights_, agent)
            info['loss'].append(loss)
            grad = autograd.grad(loss, agent._get_params(), create_graph=True)
            grads.append(grad)
        return grads, info
    

    def end_epoch(self):
        """Complete the current epoch by computing gradients using them to
        update the agent(s) parameters.

        returns -- dict of batch/epoch info
        """
        batch_obs, batch_acts = [], []
        for idx in range(self.n_players):
            batch_obs_, batch_acts_ = self.buffers[idx].drain()
            batch_obs.append(batch_obs_)
            batch_acts.append(batch_acts_)
        # compute loss and grads
        grads, step_info = self._step(batch_obs, batch_acts)

        # update agent action policies using computed grads, and compute grad norms
        grad_norms = []
        for idx in range(len(self.agents)):
            self.agents[idx].update(grads[idx])
            norms = [float(torch.norm(grad)) for grad in grads[idx]]
            grad_norms.append(norms)
        
        # compute batch info for logging
        mean_rets = [np.mean(self.batch_rets[i]) for i in range(len(self.agents))]
        mean_lens = [np.mean(self.batch_lens[i]) for i in range(len(self.agents))]
        loss = [float(step_info['loss'][i]) for i in range(len(step_info['loss']))]
        info = {
            'grad_norms': grad_norms,                # batch grads
            'avg_rets': mean_rets,       # batch return
            'avg_lens': mean_lens,       # batch len
            'loss': loss
        }

        # reset for new batch/epoch
        self._reset_batch_info()
        return info

    def end_episode(self):
        """
        Complete the current episode and compute the reward weights for the
        episode.
        """
        # get the cumulative episode return and length and compute episode
        # weights accordingly
        for idx in range(self.n_players):
            ep_ret, ep_len = self.buffers[idx].compute_batch_info()
            self.batch_rets[idx].append(ep_ret)
            self.batch_lens[idx].append(ep_len)
            self.batch_weights[idx] += [ep_ret] * ep_len
        

    def track_timestep(self, acts, rews):
        """
        Update the buffers with current timestep info.

        Keyword arguments:
        acts -- list containing the agent(s) actions for the current
        timestep
        rews -- list containing the computed rewards for the current
        timestep
        """
        # append timestep information to the buffer(s)
        for idx in range(self.n_players):
            self.buffers[idx].append_timestep(acts[idx], rews[idx])
    

    def track_obs(self, obs):
        """
        Append observations to the buffer.

        Keyword arguments:
        obs -- list containing the environment state/observation(s) for
        the current timestep
        """
        if self.shared_state:
            for  buffer in self.buffers:
                buffer.append_obs(obs)
        else:
            for idx, buffer in enumerate(self.buffers):
                buffer.append_obs(obs[idx])
    

    def batch_over(self):
        """Return whether or not a buffer's length is longer than the maximum
        batch size. Indicating the batch is over.
        """
        buff_len = max([len(self.buffers[i]._obs) for i in range(self.n_players)])
        return buff_len >= self.batch_size


    def get_agent_params(self, copy: bool=True):
        """Return a list of the current neural network parameters for all
        agents
        
        Keyword arguments:
        copy -- whether or not to return a copy of the parameters
        
        returns -- list of all agent nn parameters"""
        params = []
        for agent in self.agents:
            if copy:
                # FIXME make sure this copies as intended
                params.append(list(agent._get_params()))
            else:
                params.append(agent._get_params())
        return params
