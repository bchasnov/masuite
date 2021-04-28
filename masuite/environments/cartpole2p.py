import numpy as np
from masuite.environments.base import Environment
from masuite.environments.cartpole import CartPoleEnv


class Cartpole2PEnv(Environment):
    def __init__(self, mapping_seed, masuite_num_episodes, is_uncoupled=True):
        self.n_players = 2
        self.is_uncoupled = is_uncoupled
        self.envs = [CartPoleEnv(mapping_seed) for _ in range(self.n_players)]


    def step(self, acts, weight=-0.00):
        obs, raw_rews, done, info = [], [], [], []
        for idx in range(self.n_players):
            act = acts[idx]
            obs_, rew_, done_, info_ = self.envs[idx].step(act)
            obs.append(obs_)
            raw_rews.append(rew_)
            done.append(done_)
            info.append(info_)
        obs = np.array(obs)

        rews = None
        if self.is_uncoupled:
            rews = raw_rews
        else:
            xs = [1, -1]
            rews = [
                rews[i] + weight*self.envs[i].state[0]-xs[i]**2
                for i in range(self.n_players)
            ]
        # rews = [rews[0] + weight*(self.env1.state[0]-x1)**2, 
            #    rew2 + weight*(self.env2.state[0]-x2)**2]
        
        info = dict(p1=info[0], p2=info[1])
        done = done[0] and done[1]

        return obs, rews, info, done


    def reset(self):
        obs = [self.envs[i].reset() for i in range(self.n_players)]
        return obs

if __name__ == '__main__':
    env = Cartpole2PEnv(0, 5000)
    print(env.reset())
    print(env.step(acts=[0, 1]))