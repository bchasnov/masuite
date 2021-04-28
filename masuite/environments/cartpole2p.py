import numpy as np
from masuite.environments.base import Environment
from masuite.environments.cartpole import CartPoleEnv


class Cartpole2PEnv(Environment):
    def __init__(self, mapping_seed, masuite_num_episodes, is_uncoupled=True):
        super(Cartpole2PEnv, self).__init__()
        self.n_players = 2
        self.viewer = None
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
    

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.envs[0].x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.envs[0].length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cart2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans1 = rendering.Transform()
            self.carttrans2 = rendering.Transform()
            cart1.add_attr(self.carttrans1)
            cart2.add_attr(self.carttrans2)
            self.viewer.add_geom(cart1)
            self.viewer.add_geom(cart2)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole1.set_color(.8, .6, .4)
            pole2.set_color(.8, .6, .4)
            self.poletrans1 = rendering.Transform(translation=(0, axleoffset))
            self.poletrans2 = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.poletrans1)
            pole1.add_attr(self.carttrans1)
            pole2.add_attr(self.poletrans2)
            pole2.add_attr(self.carttrans2)
            self.viewer.add_geom(pole1)
            self.viewer.add_geom(pole2)
            self.axle1 = rendering.make_circle(polewidth/2)
            self.axle2 = rendering.make_circle(polewidth/2)
            self.axle1.add_attr(self.poletrans1)
            self.axle1.add_attr(self.carttrans1)
            self.axle2.add_attr(self.poletrans2)
            self.axle2.add_attr(self.carttrans2)
            self.axle1.set_color(.5, .5, .5)
            self.axle2.set_color(.5, .5, .5)
            self.viewer.add_geom(self.axle1)
            self.viewer.add_geom(self.axle2)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geoms = [pole1, pole2]

            if self.envs[0].state is None or self.envs[1].state is None:
                return None
            
            pole1, pole2 = self._pole_geoms[0], self._pole_geoms[1]
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pol1.v = [(l, b), (l, t), (r, t), (r, b)]
            pol2.v = [(l, b), (l, t), (r, t), (r, b)]

            x1, x2 = self.envs[0].state, self.envs[1].state
            cart1x = x1[0] * scale + screen_width / 2
            cart2x = x2[0] * scale + screen_width / 2
            self.carttrans1.set_translation(cart1x, carty)
            self.carttrans2.set_translation(cart2x, carty)
            self.poletrans1.set_rotation(-x1[2])
            self.poletrans2.set_rotation(-x2[2])

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
            

if __name__ == '__main__':
    env = Cartpole2PEnv(0, 5000)
    print(env.reset())
    print(env.step(acts=[0, 1]))
    while True:
        env.render()