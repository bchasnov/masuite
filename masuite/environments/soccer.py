import numpy as np

from masuite.environments.base import DiscreteEnvironment


class SoccerEnv(DiscreteEnvironment):
    mapping_seed = None
    n_players = 2
    env_dim = [3]
    n_acts = 5
    shared_state = True
    max_episode_len = 1000

    def __init__(self, mapping_seed: int=None):
        self.mapping_seed = mapping_seed
        self.action_space = [-5, 5, -1, 1, 0] # up, down left, right, stay
        self.A_goal = [5, 10]
        self.B_goal = [9, 14]
        self.discount_factor = 0.9
        self.goal_reward = 1

        self.episode_len = 0
    
    def seed(self):
        np.random.seed(self.mapping_seed)

    def _get_state(self):
        return (self.A_pos, self.B_pos, self.A_has_ball)
    
    def _compute_reward(self):
        curr_discount = self.discount_factor**self.episode_len
        if self.A_has_ball:
            if self.A_pos in self.A_goal:
                return curr_discount * self.goal_reward
            if self.A_pos in self.B_goal:
                return -curr_discount * self.goal_reward
        else:
            if self.B_pos in self.B_goal:
                return -curr_discount * self.goal_reward
            if self.B_pos in self.A_goal:
                return curr_discount * self.goal_reward
        return 0
    
    def _move_player(self, position, action):
        new_pos = position + action
        if new_pos < 0 or new_pos > 19:
            return position
        return new_pos
    
    def _move_A(self, A_act):
        act = self.action_space[A_act]
        new_pos = self._move_player(self.A_pos, act)
        if new_pos != self.B_pos:
            self.A_pos = new_pos
        elif self.A_has_ball:
            self.A_has_ball = False
    
    def _move_B(self, B_act):
        act = self.action_space[B_act]
        new_pos = self._move_player(self.B_pos, act)
        if new_pos != self.A_pos:
            self.B_pos = new_pos
        elif not self.A_has_ball:
            self.A_has_ball = True

    def reset(self):
        self.episode_len = 0
        # NOTE: initial position is a square from middle column of grid
        self.A_pos, self.B_pos = np.random.choice([2, 7, 12, 17], size=2, replace=False)
        self.A_has_ball = np.random.choice([True, False])
        return self._get_state()
    
    def step(self, acts):
        assert len(acts) == 2
        if np.random.random() > 0.5:
            self._move_A(acts[0])
            self._move_B(acts[1])
        else:
            self._move_B(acts[1])
            self._move_A(acts[0])
        
        reward = self._compute_reward()
        done = not reward == 0 or self.episode_len >= self.max_episode_len
        # if done:
        #     if reward != 0:
        #         print(f"Reward is {reward}, episode length is {self.episode_len}")
        #     elif self.episode_len >= self.max_episode_len:
        #         print("Reached max ep len")
        self.episode_len += 1

        return self._get_state(), [reward, -reward], done, {}