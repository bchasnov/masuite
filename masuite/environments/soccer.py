# Soccer class resembles an enviroment in OpenAI Gym
# Use reset() to initiate an episode
# Use step(actionA, actionB) to simulate an action which returns next stata, reward and isFinished
# Use render() to draw the current state
# self.action_space: num of actions
# self.state_space: <num of variabel1, num of variable2, num of variable3>

# the field is a 4x5 grid
# number the grid as
# 0,  1,  2,  3,  4
# 5,  6,  7,  8,  9
# 10, 11, 12, 13, 14,
# 15, 16, 17, 18, 19
# states are position of A, position of B and whether A or B has the ball
# actions for both A and B are (N,S,E,W,stick) which is represented as 0~5

import numpy as np

from masuite.environments.base import DiscreteEnvironment

GOAL_REWARD = 100
BALL_REWARD = 10

class SoccerEnv(DiscreteEnvironment):
    episode_length = 0
    max_episode_len = 1000
    mapping_seed = None
    n_players = 2
    env_dim = [84]
    n_acts = 5
    shared_state = True


    def __init__(self, mapping_seed=None):
        self.mapping_seed = mapping_seed
        self.action_space = [-5, 5, -1, 1, 0] # up, down left, right, stay
        self.discount_factor = 0.9
        self.episode_len = 0
    
    def seed(self):
        np.random.seed(self.mapping_seed)

    def _get_state(self):
        return (self.A_pos, self.B_pos, self.A_has_ball, self.A_goal, self.B_goal)

    def _encode_state(self):
        state = np.zeros((4, 21)) # ???
        
        # Encode Goal Locations
        state[0][self.A_goal[0]] = 1
        state[1][self.A_goal[0]] = 1
        state[2][self.B_goal[0]] = 1
        state[3][self.B_goal[0]] = 1
        state[0][self.A_goal[1]] = 1
        state[1][self.A_goal[1]] = 1
        state[2][self.B_goal[1]] = 1
        state[3][self.B_goal[1]] = 1
        
        # Encode Positions and Ball Ownership
        if self.A_has_ball:
            state[0][0] = 1
            state[1][self.A_pos+1] += 1
            state[2][self.B_pos+1] += 1
            state[3][0] = 1
        else:
            state[0][self.A_pos+1] = +1
            state[1][0] = 1
            state[2][0] = 1
            state[3][self.B_pos+1] = +1
        return state.flatten()
    
    # returns the reward for A, the reward for B is the negative by definition of zero sum game
    def _compute_reward(self):
        curr_discount = self.discount_factor**self.episode_len # As time goes on reduce reward
        goal = False # If a goal then stop the game, for some reason stops errors
        vert_factor = 0 # Encourage being in the center
        if self.A_has_ball:
            
            # Rewards for Goals
            if self.A_pos in self.A_goal:
                goal = True
                return curr_discount * -GOAL_REWARD, goal
            if self.A_pos in self.B_goal:
                goal = True
                return curr_discount * GOAL_REWARD, goal
            
            # Rewards for Positioning
            else:
                if self.A_pos in [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]:
                    vert_factor = -2
                if self.B_goal[0] % 5 == 0:
                    return (1+curr_discount) * (-(self.A_pos % 5) + BALL_REWARD + vert_factor), goal
                else:
                    return (1+curr_discount) * ((self.A_pos % 5) - (self.B_goal[0] % 5) + BALL_REWARD + vert_factor), goal
        else:
            if self.B_pos in self.B_goal:
                goal = True
                return curr_discount * GOAL_REWARD, goal
            if self.B_pos in self.A_goal:
                goal = True
                return curr_discount * -GOAL_REWARD, goal
            else:
                if self.B_pos in [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]:
                    vert_factor = 2
                if self.A_goal[0] % 5 == 0:
                    return (1+curr_discount) * ((self.B_pos % 5) - BALL_REWARD + vert_factor), goal
                else:
                    return (1+curr_discount) * (-(self.B_pos % 5) + (self.A_goal[0] % 5) - BALL_REWARD + vert_factor), goal
        return 0, goal
    
    # If they are moving out of bounds, then they don't move.
    def _move_player(self, position, action):
        new_pos = position + self.action_space[action]
        if new_pos < 0 or new_pos > 19 or (position in [0, 5, 10, 15] and self.action_space[action] == -1) or (position in [4, 9, 14, 19] and self.action_space[action] == 1):
            return position

        
        return new_pos
    
    def _move_A(self, A_act):
        A_new_pos = self._move_player(self.A_pos, A_act)
        if A_new_pos != self.B_pos:
            self.A_pos = A_new_pos
        # If A run into B with a ball, give the ball to B, and vice versa
        elif self.A_has_ball:
            self.A_has_ball = False
        elif not self.A_has_ball:
            self.A_has_ball = True
    
    def _move_B(self, B_act):
        B_new_pos = self._move_player(self.B_pos, B_act)
        if B_new_pos != self.A_pos:
            self.B_pos = B_new_pos
        # If B run into A with a ball, give the ball to A, and vice versa
        elif not self.A_has_ball:
            self.A_has_ball = True
        elif self.A_has_ball:
            self.A_has_ball = False
            
    def seed(self, seed=None):
        np.random.seed(seed)
        
    def reset(self):
        self.episode_len = 0
        # NOTE: initial position is a square from middle column of grid
        self.A_pos= np.random.choice([6, 11, 8, 13], replace=False)
        # if (self.A_pos == 6 or self.A_pos == 11):
        #     self.B_pos = np.random.choice([8, 13], replace=False)
        # else:
        #     self.B_pos = np.random.choice([6, 11], replace=False)
        
        # Set Positions
        if (self.A_pos == 6):
            self.B_pos = 13
        elif (self.A_pos == 11):
            self.B_pos = 8
        elif (self.A_pos == 8):
            self.B_pos = 11
        else:
            self.B_pos = 6
        
        # Set Goals based on initial positions
        if (self.A_pos == 6 or self.A_pos == 11):
            self.A_goal = [5, 10]
            self.B_goal = [9, 14]
        else:
            self.A_goal = [9, 14]
            self.B_goal = [5, 10]
            
        # Set Initial Ball Ownership
        self.A_has_ball = np.random.choice([True, False])
        #print("****************************************** RESET ******************************************")
        return self._encode_state()
    
    def step(self, acts):
        assert len(acts) == 2
        
        # SIMGRAD
        if np.random.random() > 0.5:
            self._move_A(acts[0])
            self._move_B(acts[1])
        else:
            self._move_B(acts[1])
            self._move_A(acts[0])
            
        # STACKGRAD
        # self._move_A(acts[0])
        # self._move_B(acts[1])

        reward, goal = self._compute_reward()
        done = False
        if (goal) or (self.episode_len >= self.max_episode_len):
            done = True
        # if done:
        #     if reward != 0:
        #         print(f"Reward is {reward}, episode length is {self.episode_len}")
        #     elif self.episode_len >= self.max_episode_len:
        #         print("Reached max ep len")
        self.episode_len += 1

        return self._encode_state(), [reward, -reward], done, {}