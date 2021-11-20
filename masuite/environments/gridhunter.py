from masuite.environments.base import Environment
import numpy as np
import math

class GridHunterEnv(Environment):
    n_acts = 4
    mapping_seed = None
    n_players = 2
    env_dim = [6]
    shared_state = True
    episode_length = 0
    max_episode_length = 500

    def __init__(self, mapping_seed: None):
        self.mapping_seed = mapping_seed
        self.actions = [0, 1, 2, 3]
        self.episode_len = 0
        self.map_size = 25
        self.MAX_REWARD = self.dist((1, 1), (self.map_size - 1, self.map_size - 1))
        self.occupancy = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):  # sets borders of grid
            self.occupancy[0][i] = 1
            self.occupancy[self.map_size - 1][i] = 1
            self.occupancy[i][0] = 1
            self.occupancy[i][self.map_size - 1] = 1
        self.h1_pos = [self.map_size - 3, 1]  # one above of bottom left corner
        self.h2_pos = [self.map_size - 2, 2]  # one right of bottom left corner
        self.p_pos = [1, self.map_size - 2]  # top right corner

    def step(self, acts):
        # reward based on distance from prey of closest agent
        act1, act2 = acts
        self.p_pos = self.prey_act()
        self.h1_pos = self.hunter_act(self.h1_pos, act1)
        self.h2_pos = self.hunter_act(self.h2_pos,act2)
        self.episode_length += 1

        reward1 = (self.MAX_REWARD - self.dist(self.h1_pos, self.p_pos)) * 0.01
        reward2 = (self.MAX_REWARD - self.dist(self.h2_pos, self.p_pos)) * 0.01

        if self.h1_pos == self.p_pos:
            reward1 = self.MAX_REWARD
        if self.h2_pos == self.p_pos:
            reward2 = self.MAX_REWARD

        return self.temp_state(), [reward1, reward2], \
               reward1 >= self.MAX_REWARD or reward2 >= self.MAX_REWARD \
               or self.episode_length >= self.max_episode_length, {}

    def reset(self):
        self.episode_length = 0
        self.occupancy = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):  # sets borders of grid
            self.occupancy[0][i] = 1
            self.occupancy[self.map_size - 1][i] = 1
            self.occupancy[i][0] = 1
            self.occupancy[i][self.map_size - 1] = 1
        self.h1_pos = [self.map_size - 3, 1]  # one above of bottom left corner
        self.h2_pos = [self.map_size - 2, 2]  # one right of bottom left corner
        self.p_pos = [1, self.map_size - 2]  # top right corner
        return self.temp_state() 

    def prey_act(self):
        # responds to previous hunter position in order to maximize distance from closest hunter
        counter = 0
        possible_pos = [self.p_pos, self.p_pos, self.p_pos, self.p_pos]
        if self.occupancy[self.p_pos[0] - 1][self.p_pos[1]] != 1:
            possible_pos[counter] = (self.p_pos[0] - 1, self.p_pos[1])
            counter += 1
        if self.occupancy[self.p_pos[0] + 1][self.p_pos[1]] != 1:
            possible_pos[counter] = (self.p_pos[0] + 1, self.p_pos[1])
            counter += 1
        if self.occupancy[self.p_pos[0]][self.p_pos[1] - 1] != 1:
            possible_pos[counter] = (self.p_pos[0], self.p_pos[1] - 1)
            counter += 1
        if self.occupancy[self.p_pos[0]][self.p_pos[1] + 1] != 1:
            possible_pos[counter] = (self.p_pos[0], self.p_pos[1] + 1)
            counter += 1

        max_p_pos = self.p_pos
        for i in range(counter):  # if the closest hunter is farther away
            if (min(self.dist(self.h1_pos, possible_pos[counter]), self.dist(self.h2_pos, possible_pos[counter]))) > (min(self.dist(self.h1_pos, max_p_pos), self.dist(self.h2_pos, max_p_pos))):
                max_p_pos = possible_pos[counter]
        return max_p_pos

    def hunter_act(self, pos, act):
        # moves according to given action and occupancy
        if act == 0:  # move up
            if self.occupancy[pos[0] - 1][pos[1]] != 1:  # if can move
                pos[0] = pos[0] - 1
        elif act == 1:  # move down
            if self.occupancy[pos[0] + 1][pos[1]] != 1:  # if can move
                pos[0] = pos[0] + 1
        elif act == 2:  # move left
            if self.occupancy[pos[0]][pos[1] - 1] != 1:  # if can move
                pos[1] = pos[1] - 1
        elif act == 3:  # move right
            if self.occupancy[pos[0]][pos[1] + 1] != 1:  # if can move
                pos[1] = pos[1] + 1
        return pos

    def dist(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def seed(self, seed=None):
        np.random.seed()

    def temp_state(self):
        state = np.zeros((1, 6))
        state[0, 0] = self.h1_pos[0] / self.map_size
        state[0, 1] = self.h1_pos[1] / self.map_size
        state[0, 2] = self.h2_pos[0] / self.map_size
        state[0, 3] = self.h2_pos[1] / self.map_size
        state[0, 4] = self.p_pos[0] / self.map_size
        state[0, 5] = self.p_pos[1] / self.map_size
        return state