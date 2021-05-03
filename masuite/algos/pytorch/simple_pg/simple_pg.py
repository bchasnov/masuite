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
        # self.buffer = Buffer(
        #     obs_dim=obs_dim,
        #     act_dim=act_dim,
        #     n_players=n_players,
        #     max_batch_len=batch_size
        # )
        self.batch_rets = []
        self.batch_lens = []
        self.batch_weights = []
        for _ in range(self.n_players):
            self.batch_rets.append([])
            self.batch_lens.append([])
            self.batch_weights.append([])
    
    def _compute_loss(self, obs, act, rews, agent):
        # print('policy: ', agent._get_policy(obs).sample())
        # import pdb
        # pdb.set_trace()
        logp = agent._get_policy(obs).log_prob(act)
        # print('logp: ', logp)
        return -(logp * rews).mean()
    
    
    def step(self, obs, acts):
        # print(obs.shape, acts.shape)
        info = {
            'loss': [],
        }
        grads = []
        obs = torch.as_tensor(obs, dtype=torch.float32)
        acts = torch.as_tensor(acts, dtype=torch.float32)
        weights = torch.as_tensor(self.batch_weights, dtype=torch.float32)
        # print('obs: ', obs.shape)
        # print('acts: ', acts.shape)
        # print(acts)
        # print('rews: ', rews.shape)
        logps = []
        for idx in range(len(self.agents)):
            agent = self.agents[idx]
            loss = self._compute_loss(obs[idx], acts[idx], weights, agent)
            info['loss'].append(loss)
            grad = autograd.grad(loss, agent._get_params(), create_graph=True)
            grads.append(grad)
        # print('grads: ', grads)
        # exit()
        return grads, info


    def update(self, obs, acts, rews, done):
        # append timestep information to the buffer(s)
        if self.shared_state:
            for idx in range(self.n_players):
                self.buffers[idx].append_timestep(obs, acts[idx], rews[idx])
        else:
            for idx in range(self.n_players):
                self.buffers[idx].append_timestep(obs[idx], acts[idx], rews[idx])

        if done:
            for idx in range(self.n_players):
                ep_ret, ep_len = self.buffers[idx].compute_batch_info()
                self.batch_rets[idx].append(ep_ret)
                self.batch_lens[idx].append(ep_len)
                self.batch_weights[idx] += [ep_ret] * ep_len


            buff_lens = [len(self.buffers[i]._obs) for i in range(self.n_players)]
            if max(buff_lens) > self.batch_size:
                batch_obs, batch_acts = [], []
                for idx in range(self.n_players):
                    batch_obs_, batch_acts_ = self.buffers[idx].drain()
                    batch_obs.append(batch_obs_)
                    batch_acts.append(batch_acts_)
                grads, step_info = self.step(batch_obs, batch_acts)
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
