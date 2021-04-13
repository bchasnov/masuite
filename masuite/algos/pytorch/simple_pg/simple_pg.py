import torch
import torch.autograd as autograd
from masuite.algos.utils.buffer import Buffer

class SimplePG:
    def __init__(self, agents, obs_dim, act_dim, n_players, batch_size=5000):
        self.agents = agents
        self.buffer = Buffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_players=n_players,
            max_batch_len=batch_size
        )
        self.batch_rets = []
        self.batch_lens = []
        self.batch_weights = []
        self.batch_size = batch_size
    
    def _compute_loss(self, obs, act, weights, agent):
        logp = agent._get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    
    
    def step(self, obs, acts):
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
            loss = self._compute_loss(obs, acts[idx], weights[idx], agent)
            info['loss'].append(loss)
            grad = autograd.grad(loss, agent._get_params(), create_graph=True)
            grads.append(grad)
        return grads, info


    def update(self, obs, acts, rews, done):
        self.buffer.append_timestep(obs, acts, rews)
        if done:
            ep_ret, ep_len = self.buffer.compute_batch_info()

            # the weight for each logprob(a|s) is R(tau)
            if self.batch_weights == []:
                for idx in range(len(self.agents)):
                    self.batch_weights.append([ep_ret[idx]]*ep_len)
            else:
                for idx in range(len(self.agents)):
                    self.batch_weights[idx] += [ep_ret[idx]] * ep_len

            if len(self.buffer._obs) > self.batch_size:
                batch_obs, batch_acts = self.buffer.drain()
                grads, info = self.step(batch_obs, batch_acts)
                for idx in range(len(self.agents)):
                    self.agents[idx].update(grads[idx])
                return info['loss'], ep_ret, ep_len
        return None, None, None
