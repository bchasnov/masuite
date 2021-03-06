# Soccer class resembles an enviroment in OpenAI Gym
# Use reset() to initiate an episode
# Use step(actionA, actionB) to simulate an action which returns next stata, reward and isFinished
# Use render() to draw the current state
# self.action_space: num of actions
# self.state_space: <num of variabel1, num of variable2, num of variable3>

# the fild is a 2x4 grid
# number the grid as
# 0, 1, 2, 3
# 4, 5, 6, 7
# states are position of A, position of B and whether A or B has the ball
# actions for both A and B are (N,S,E,W,stick) which is represented as 0~4
import numpy as np
from masuite.environments.base import DiscreteEnvironment
GOAL_REWARD = 100

class SmallSoccerEnv(DiscreteEnvironment):
    max_episode_len = 1000
    mapping_seed = None
    n_players = 2
    env_dim = [36]
    n_acts = 5
    shared_state = True

    def __init__(self, mapping_seed=None):
        self.mapping_seed = mapping_seed
        self.actions = [-4, 4, 1, -1, 0]
        self.state_space = (8, 8, 2)

        self.discount_factor = 0.9
        self.episode_len = 0

    def __showCurrentState(self):
        return (self.posOfA, self.posOfB, self.AHasBall)

    def __encodeCurrentState(self):
        state = np.zeros((4, 9))
        if self.AHasBall:
            state[0][0] = 1
            state[1][self.posOfA+1] = 1
            state[2][self.posOfB+1] = 1
            state[3][0] = 1
        else:
            state[0][self.posOfA+1] = 1
            state[1][0] = 1
            state[2][0] = 1
            state[3][self.posOfB+1] = 1
        return state.flatten()

    # returns the reward for A, the reward for B is the negative by definition of zero sum game
    def __calculateReward(self):
        curr_discount = self.discount_factor**self.episode_len
        if self.AHasBall:
            if self.posOfA == 0 or self.posOfA == 4:
                return curr_discount * GOAL_REWARD
            if self.posOfA == 3 or self.posOfA == 7:
                return curr_discount * -GOAL_REWARD
        else:
            if self.posOfB == 0 or self.posOfB == 4:
                return curr_discount * GOAL_REWARD
            if self.posOfB == 3 or self.posOfB == 7:
                return curr_discount * -GOAL_REWARD
        return 0

    # calculate the postion of a player after a move
    # player sticks if moving towards a wall
    def __movePlayer(self, postion, action):
        newPostion = postion + self.actions[action]
        if newPostion < 0 or newPostion > 7:
            return postion
        else:
            return newPostion

    def __moveA(self, actionOfA):
        newPosOfA = self.__movePlayer(self.posOfA, actionOfA)
        if newPosOfA != self.posOfB:
            self.posOfA = newPosOfA
        # if A run into B with a ball, give the ball to B
        elif self.AHasBall:
            self.AHasBall = False

    def __moveB(self, actionOfB):
        newPosOfB = self.__movePlayer(self.posOfB, actionOfB)
        if newPosOfB != self.posOfA:
            self.posOfB = newPosOfB
        # if B run into A with a ball, give the ball to A
        elif not self.AHasBall:
            self.AHasBall = True
    

    def seed(self, seed=None):
        np.random.seed(seed)


    # initilized game with random ball poccession
    def reset(self):
        self.episode_len = 0
        self.posOfA, self.posOfB = np.random.choice([1,2,5,6], size=2, replace=False)
        self.AHasBall = np.random.choice([True, False])
        return self.__encodeCurrentState()

    # take a step in the game given actions of A and B
    # return next state, reward and whether the game is dones
    def step(self, acts):
        # Note: action input is index of action to take
        assert len(acts) == 2
        if np.random.random() > 0.5:
            # A moves first
            self.__moveA(acts[0])
            self.__moveB(acts[1])
        else:
            # B moves first
            self.__moveB(acts[1])
            self.__moveA(acts[0])

        reward = self.__calculateReward()
        done = not reward == 0 or self.episode_len >= self.max_episode_len
        self.episode_len += 1
        return self.__encodeCurrentState(), [reward, -reward], done, {}

    def render(self):
        out = "---------------------\n"
        for i in range(2):
            for j in range(4):
                position = i * 4 + j
                if self.posOfA == position:
                    if self.AHasBall:
                        out += "| A* "
                    else:
                        out += "| A  "
                elif self.posOfB == position:
                    if not self.AHasBall:
                        out += "| B* "
                    else:
                        out += "| B  "
                else:
                    out += "|    "
            out += "|\n---------------------\n"
        print(out)