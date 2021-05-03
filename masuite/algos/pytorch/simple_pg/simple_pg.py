import torch
import torch.autograd as autograd
from masuite.algos.utils.buffer import SingleBuffer

class SimplePG:
    def __init__(self,
        agents,
        obs_dim,
        act_dim,
        shared_state,
        n_players,
        batch_size=5000
    ):
        self.agents = agents
        self.shared_state = shared_state
        self.n_players = n_players
        self.batch_size = batch_size
        self.buffers = [SingleBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            max_batch_len=batch_size
        ) for _ in range(n_players)]
        self.batch_rets = []
        self.batch_lens = []
        self.batch_weights = []
        for _ in range(self.n_players):
            self.batch_rets.append([])
            self.batch_lens.append([])
            self.batch_weights.append([])
    
    def _compute_loss(self, obs, act, rews, agent):
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
        return -(logp * rews).mean()
    
    
    def _step(self, obs, acts):
        """Compute the batch loss and take a single gradient descent
        optimization step for each agent.

        Keyword arguments:
        obs -- list of batch observations
        acts -- list of batch actions taken
        """
        info = {
            'loss': [],
        }
        grads = []
        obs = torch.as_tensor(obs, dtype=torch.float32)
        acts = torch.as_tensor(acts, dtype=torch.float32)
        weights = torch.as_tensor(self.batch_weights, dtype=torch.float32)
        logps = []
        for idx in range(len(self.agents)):
            agent = self.agents[idx]
            loss = self._compute_loss(obs[idx], acts[idx], weights, agent)
            info['loss'].append(loss)
            grad = autograd.grad(loss, agent._get_params(), create_graph=True)
            grads.append(grad)
        return grads, info


    def _end_episode(self):
        """
        Complete the current episode and compute the reward weights for the
        episode.


        returns -- None if batch has not ended, dict of batch info if the
        batch is over
        """
        # get the cumulative episode return and length and compute episode
        # weights accordingly
        for idx in range(self.n_players):
            ep_ret, ep_len = self.buffers[idx].compute_batch_info()
            self.batch_rets[idx].append(ep_ret)
            self.batch_lens[idx].append(ep_len)
            self.batch_weights[idx] += [ep_ret] * ep_len


        # check the current buffer sizes to check if the batch is over
        buff_len = max([len(self.buffers[i]._obs) for i in range(self.n_players)])
        if buff_len > self.batch_size:
            batch_obs, batch_acts = [], []
            for idx in range(self.n_players):
                batch_obs_, batch_acts_ = self.buffers[idx].drain()
                batch_obs.append(batch_obs_)
                batch_acts.append(batch_acts_)
            grads, step_info = self._step(batch_obs, batch_acts)
            for idx in range(len(self.agents)):
                self.agents[idx].update(grads[idx])
            info = {
                'loss': step_info['loss'],
                'batch_rets': self.batch_rets,
                'batch_lens': self.batch_lens
            }
            self.batch_rets, self.batch_lens, self.batch_weights = [], [], []
            for _ in range(self.n_players):
                self.batch_rets.append([])
                self.batch_lens.append([])
                self.batch_weights.append([])
            return info
        return None



    def update(self, obs, acts, rews, done):
        """
        Update the buffers with environment information and end the current
        episode if neeeded.


        Keyword arguments:
        obs -- list containing the environment state/observation(s) for
        the current timestep
        acts -- list containing the agent(s) actions for the current
        timestep
        rews -- list containing the computed rewards for the current
        timestep
        done -- bool indicating whether the current environment episode
        has ended

        returns -- None if batch is not over, dict of batch info if the
        batch has ended.
        """
        # append timestep information to the buffer(s)
        if self.shared_state:
            for idx in range(self.n_players):
                self.buffers[idx].append_timestep(obs, acts[idx], rews[idx])
        else:
            for idx in range(self.n_players):
                self.buffers[idx].append_timestep(obs[idx], acts[idx], rews[idx])

        if done:
            return self._end_episode()
        
        return None
