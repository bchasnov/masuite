import torch
import torch.autograd as autograd

class SimplePG:
    def __init__(self, agents):
        self.agents = agents
    
    def _compute_loss(self, obs, act, weights, agent):
        logp = self.agent._get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    
    def step(self, acts, obs, weights):
        info = dict(
            'loss': [],
            'grad': []
        )
        obs = torch.as_tensor(obs, dtype=torch.float32)
        acts = torch.as_tensor(acts, dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)
        logps = []
        for idx in range(len(agents)):
            agent = self.agents[idx]
            loss = self._compute_loss(obs, acts[idx], weights, agent)
            info['loss'].append(loss)
            grad = autograd.grad(loss, agent._get_params(), create_graph=True)
            info['grad'].append(grad)
            agent.update(grad)
        return info
        